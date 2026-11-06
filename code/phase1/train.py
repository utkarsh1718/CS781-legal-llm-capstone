"""
train.py -- Phase-1 pre-training loop (single GPU or DDP on Kaggle 2xT4).
Launch:  torchrun --standalone --nproc_per_node=2 /kaggle/working/src/train.py --config <run.json>
fp16 autocast + GradScaler | DDP + grad accumulation (no_sync) | torch.compile | WSD or cosine LR
| curriculum mixture | per-source val loss + bits/word | atomic latest.pt / best.pt | exact
auto-resume | clean stop before the Kaggle session limit | per-step model hook (MoE balancing).
"""
from __future__ import annotations
import argparse, csv, json, math, os, sys, time
from contextlib import nullcontext
import torch
import torch.distributed as dist

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data import ShardSampler, fixed_val_windows, mixture_weights, SOURCES   # noqa: E402
from lm_common import count_params                                         # noqa: E402


def build_model(arch: str, model_cfg: dict):
    """Model registry (M3, M4 added later)."""
    if arch == "dense":
        from model_dense import DenseGPT, DenseConfig
        return DenseGPT(DenseConfig(**model_cfg))
    if arch == "mla_moe":
        from model_mla_moe import MLAMoEGPT, MLAMoEConfig
        return MLAMoEGPT(MLAMoEConfig(**model_cfg))
    if arch == "encdec":
        from model_encdec import EncDecLM, EncDecConfig
        return EncDecLM(EncDecConfig(**model_cfg))
    if arch == "copy":
        from model_copy import CopyGPT, CopyConfig
        return CopyGPT(CopyConfig(**model_cfg))
    raise ValueError(f"unknown arch {arch}")


def lr_at(step, total, cfg):
    """Warmup, then WSD (constant -> cosine decay over the last decay_frac) or plain cosine."""
    peak, floor = cfg["lr"], cfg["lr"] * cfg["min_lr_frac"]
    warm = max(1, int(cfg["warmup_frac"] * total))
    if step < warm:
        return peak * (step + 1) / warm
    if cfg["schedule"] == "cosine":
        prog = (step - warm) / max(1, total - warm)
        return floor + 0.5 * (peak - floor) * (1 + math.cos(math.pi * prog))
    decay_start = int((1 - cfg["decay_frac"]) * total)
    if step < decay_start:
        return peak
    prog = (step - decay_start) / max(1, total - decay_start)
    return floor + 0.5 * (peak - floor) * (1 + math.cos(math.pi * prog))


@torch.no_grad()
def evaluate(model, val_windows, tokens_per_word, device, micro_bs):
    """Mean loss per source on fixed windows -> loss, perplexity, bits per word."""
    model.eval()
    res = {}
    for s, W in val_windows.items():
        tot, n = 0.0, 0
        for i in range(0, len(W), micro_bs):
            b = W[i:i + micro_bs].to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.float16):
                loss = model(b[:, :-1], b[:, 1:])
            tot += loss.item() * len(b); n += len(b)
        nats = tot / max(1, n)
        res[s] = {"loss": nats, "ppl": math.exp(min(nats, 20)),
                  "bits_per_word": nats * tokens_per_word.get(s, 1.3) / math.log(2)}
    model.train()
    return res


def atomic_save(obj, path):
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)                     # never leaves a half-written checkpoint


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    cfg = json.load(open(ap.parse_args().config))

    # ---- distributed setup -----------------------------------------------------------------
    ddp = int(os.environ.get("WORLD_SIZE", "1")) > 1
    if ddp:
        dist.init_process_group("nccl")
        rank, world = dist.get_rank(), dist.get_world_size()
        local = int(os.environ["LOCAL_RANK"])
    else:
        rank, world, local = 0, 1, 0
    torch.cuda.set_device(local)
    device = torch.device("cuda", local)
    master = rank == 0
    torch.manual_seed(cfg["seed"])
    out_dir = cfg["out_dir"]
    os.makedirs(out_dir, exist_ok=True)
    log = (lambda *a: print(*a, flush=True)) if master else (lambda *a: None)

    # ---- batch geometry --------------------------------------------------------------------
    T, mbs = cfg["seq_len"], cfg["micro_bs"]
    accum = cfg["global_bs_seqs"] // (mbs * world)
    assert accum >= 1 and accum * mbs * world == cfg["global_bs_seqs"], "global batch must divide evenly"
    tokens_per_step = cfg["global_bs_seqs"] * T
    total_steps = cfg["max_tokens"] // tokens_per_step

    # ---- data ------------------------------------------------------------------------------
    sampler = ShardSampler(cfg["data_dir"], "train", T)
    val_windows = fixed_val_windows(cfg["data_dir"], T, cfg["val_windows"])
    tpw = {s: v["tokens_per_word"] for s, v in
           json.load(open(f"{cfg['data_dir']}/val_loss_stats.json")).items()}

    # ---- model / optimiser (weight decay on matrices only) ----------------------------------
    raw = build_model(cfg["arch"], cfg["model"]).to(device)
    n_params = count_params(raw)
    n_active = raw.active_params() if hasattr(raw, "active_params") else n_params
    decay = [p for n, p in raw.named_parameters() if p.dim() >= 2]
    no_decay = [p for n, p in raw.named_parameters() if p.dim() < 2]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": cfg["weight_decay"]},
                             {"params": no_decay, "weight_decay": 0.0}],
                            lr=cfg["lr"], betas=tuple(cfg["betas"]), eps=1e-8, fused=True)
    scaler = torch.amp.GradScaler("cuda", init_scale=2.0 ** 14)

    # ---- resume ----------------------------------------------------------------------------
    start_step, tokens_seen, best_val, train_hours = 0, 0, float("inf"), 0.0
    ck_path = f"{out_dir}/latest.pt"
    if os.path.exists(ck_path):
        ck = torch.load(ck_path, map_location=device, weights_only=False)
        raw.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"]); scaler.load_state_dict(ck["scaler"])
        start_step, tokens_seen = ck["step"], ck["tokens_seen"]
        best_val, train_hours = ck.get("best_val", float("inf")), ck.get("train_hours", 0.0)
        log(f"RESUMED from step {start_step} ({tokens_seen/1e6:.0f}M tokens, {train_hours:.2f} h so far)")

    model = raw
    if cfg.get("compile", True):
        try:
            model = torch.compile(raw)
        except Exception as e:
            log("torch.compile failed, running eager:", repr(e)[:200])
    if ddp:
        # broadcast_buffers=False: MoE load counters are per-rank; router biases stay identical
        # because every rank applies the same all-reduced update in after_step()
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local], broadcast_buffers=False)

    log(f"arch={cfg['arch']} params={n_params/1e6:.1f}M (active/token {n_active/1e6:.1f}M) | world={world} "
        f"micro_bs={mbs} accum={accum} | {tokens_per_step/1e3:.0f}k tokens/step | "
        f"{total_steps} steps = {cfg['max_tokens']/1e9:.2f}B tokens")
    json.dump({**cfg, "n_params": n_params, "n_active_params": n_active, "total_steps": total_steps},
              open(f"{out_dir}/run_config.json", "w"), indent=1)
    csv_path = f"{out_dir}/train_log.csv"
    if master and not os.path.exists(csv_path):
        with open(csv_path, "w", newline="") as f:
            csv.writer(f).writerow(["step", "tokens_M", "loss", "lr", "grad_norm", "scale",
                                    "tok_per_s", "sc_frac", "hours"])

    def save_latest(step):
        if master:
            atomic_save({"model": raw.state_dict(), "opt": opt.state_dict(), "scaler": scaler.state_dict(),
                         "step": step, "tokens_seen": tokens_seen, "best_val": best_val,
                         "train_hours": train_hours + (time.time() - t_start) / 3600, "cfg": cfg}, ck_path)

    # ---- training loop ---------------------------------------------------------------------
    model.train()
    t_start = t_log = t_ck = time.time()
    tok_since_log, n_acc = 0, 0
    loss_acc = torch.zeros((), device=device)          # accumulate on GPU: no per-micro-step sync
    stop = False
    for step in range(start_step, total_steps):
        frac = step / total_steps
        lr = lr_at(step, total_steps, cfg)
        for g in opt.param_groups:
            g["lr"] = lr
        w = mixture_weights(cfg["mixture"], frac, sampler.n_tokens, 1 - cfg["decay_frac"])

        for micro in range(accum):
            x, y, _ = sampler.batch(w, mbs, cfg["seed"], step, rank, micro)
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            sync = nullcontext() if (not ddp or micro == accum - 1) else model.no_sync()
            with sync:                                  # all-reduce grads only on last micro-step
                with torch.autocast("cuda", dtype=torch.float16):
                    loss = model(x, y) / accum
                scaler.scale(loss).backward()
            loss_acc += loss.detach(); tok_since_log += x.numel() * world
        n_acc += 1
        scaler.unscale_(opt)
        gnorm = torch.nn.utils.clip_grad_norm_(raw.parameters(), cfg["grad_clip"]).item()
        scaler.step(opt); scaler.update()               # skips the step if grads overflowed
        opt.zero_grad(set_to_none=True)
        extra = raw.after_step() if hasattr(raw, "after_step") else {}   # MoE bias balancing
        tokens_seen += tokens_per_step

        # ---- logging -------------------------------------------------------------------------
        if master and ((step + 1) % cfg["log_every"] == 0 or step == start_step):
            dt = time.time() - t_log
            row = [step + 1, round(tokens_seen / 1e6, 1), round(loss_acc.item() / n_acc, 4), f"{lr:.2e}",
                   round(gnorm, 3), int(scaler.get_scale()), int(tok_since_log / dt),
                   round(w["supreme_court"], 2), round(train_hours + (time.time() - t_start) / 3600, 3)]
            with open(csv_path, "a", newline="") as f:
                csv.writer(f).writerow(row)
            log(f"step {row[0]:6d}/{total_steps} | {row[1]:7.1f}M tok | loss {row[2]:.4f} | lr {row[3]} | "
                f"gnorm {row[4]:.2f} | scale {row[5]} | {row[6]/1e3:.1f}k tok/s | SC w {row[7]} | {row[8]:.2f} h"
                + "".join(f" | {k} {v}" for k, v in extra.items()))
            t_log, tok_since_log, n_acc = time.time(), 0, 0
            loss_acc.zero_()

        # ---- evaluation + best checkpoint ------------------------------------------------------
        last = step + 1 == total_steps
        if (step + 1) % cfg["eval_every"] == 0 or last:
            if master:
                res = evaluate(raw, val_windows, tpw, device, mbs)
                sc = res["supreme_court"]["loss"]
                with open(f"{out_dir}/eval_log.jsonl", "a") as f:
                    f.write(json.dumps({"step": step + 1, "tokens_M": tokens_seen / 1e6, **res}) + "\n")
                log("  EVAL " + " | ".join(f"{s[:5]} loss {r['loss']:.3f} bpw {r['bits_per_word']:.3f}"
                                           for s, r in res.items()))
                if sc < best_val:
                    best_val = sc
                    atomic_save({"model": {k: v.half() for k, v in raw.state_dict().items()},
                                 "arch": cfg["arch"], "model_cfg": cfg["model"], "step": step + 1,
                                 "val": res}, f"{out_dir}/best.pt")
                    log(f"  new best SC val loss {sc:.4f} -> best.pt")
            if ddp:
                dist.barrier()

        # ---- periodic checkpoint + time limit (decisions all-reduced -> ranks stay in lockstep)
        elapsed_h = (time.time() - t_start) / 3600
        flags = torch.tensor([float(elapsed_h > cfg["time_limit_hours"]),
                              float((time.time() - t_ck) / 60 > cfg["ckpt_every_min"])], device=device)
        if ddp:
            dist.all_reduce(flags, op=dist.ReduceOp.MAX)
        stop, want_ck = flags[0].item() > 0, flags[1].item() > 0
        if last or stop or want_ck:
            save_latest(step + 1)
            t_ck = time.time()
            log(f"  checkpoint saved at step {step + 1}")
            if ddp:
                dist.barrier()
        if stop:
            log(f"Time limit reached ({elapsed_h:.2f} h) -> stopped cleanly at step {step + 1}; "
                f"re-run the same command to resume.")
            break

    if master and not stop:
        log(f"TRAINING COMPLETE: {tokens_seen/1e9:.2f}B tokens, best SC val loss {best_val:.4f}")
    if ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
