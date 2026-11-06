"""p7_run.py -- run a Phase-7 plan in the background:  python3 p7_run.py --plan stats/p7_x.json  [--smoke]
Plan = list of {"name": ..., "method": ..., <run() kwargs>}; finished runs (stats/p7_runs/<name>.json) are skipped."""
import argparse, json, os, sys, time, traceback

import torch

sys.path.insert(0, "/kaggle/working/src")
import p7_pcr as P                                                      # noqa: E402

ap = argparse.ArgumentParser(); ap.add_argument("--plan"); ap.add_argument("--smoke", action="store_true"); ap.add_argument("--smoke2", action="store_true")
args = ap.parse_args()


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


if args.smoke2:
    P.smoke2(log); sys.exit()
if args.smoke:
    P.smoke(log); sys.exit()
for job in json.load(open(args.plan)):
    name = job.pop("name")
    if os.path.exists(f"{P.OUT}/{name}.json"):
        log(f"skip {name} (already done)"); continue
    try:
        (P.run_kd if 'teacher' in job else P.run)(name, log=log, **job)
    except Exception:
        log(f"!!! {name} FAILED"); traceback.print_exc(); sys.stdout.flush()
    torch.cuda.empty_cache()
log("PLAN DONE")
