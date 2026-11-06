
import os, sys, time
from huggingface_hub import HfApi

api  = HfApi(token=os.environ["HF_WRITE_TOKEN"])          # token only via env
repo = f"{api.whoami()['name']}/cs781-phase1-work"
api.create_repo(repo, private=True, exist_ok=True)         # no-op if it already exists

ALLOW = [
    # ---------------- Phase 1 (unchanged) ----------------
    "src/**", "tokenizer/**", "data/**", "stats/**", "submissions/**",
    "checkpoints/*/best.pt", "checkpoints/*/config.json",
    "checkpoints/*/*.csv", "checkpoints/*/*.jsonl", "checkpoints/*/stdout.log",
    "checkpoints/m4_copy/latest.pt",
    # ---------------- Phase 2 (new) ----------------------
    "data_p2/**",            # downloaded L-NER folds + any converted external data
    "p2_ckpt/**",            # fine-tuned taggers (fp16 state dicts)
    "submissions_p2/**", "data_p5/tok/**", "data_p5/clean/**", "p5_ckpt/**", "p5_prob/**", "p5_chunks/**", "submissions_p5/**", "data_p4/tok/**", "data_p4/clean/**", "p4_ckpt/**", "p4_chunks/**", "p4_prob/**", "submissions_p4/**", "data_p3/**", "p3_ckpt/**", "p3_lp/**", "submissions_p3/**",     # the 8 test zips + prediction.json
    # ---------------- M1 ablations (new) -----------------
    "checkpoints/abl_*/*.log", "checkpoints/abl_*/*.txt",   # (best.pt/csv/jsonl already covered above)
]
IGNORE = ["**/__pycache__/**", "**/*.tmp", "**/.ipynb_checkpoints/**"]

t0 = time.time()
try:
    api.upload_folder(repo_id=repo, folder_path="/kaggle/working",
                      allow_patterns=ALLOW, ignore_patterns=IGNORE,
                      commit_message=f"backup {time.strftime('%Y-%m-%d %H:%M')}")
    print(f"BACKUP DONE -> {repo} in {time.time()-t0:.0f}s")
except Exception as e:
    print("BACKUP FAILED:", repr(e)); sys.exit(1)
