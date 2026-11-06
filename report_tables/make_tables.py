"""make_tables.py -- build the report's LaTeX result tables straight from results/stats (no hand-copied numbers).

    python report_tables/make_tables.py      # writes report_tables/phase*.tex
Tables use booktabs (\\usepackage{booktabs}); include them with \\input{report_tables/phase7.tex}.
"""
import csv, glob, json, os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ST, OUT = f"{ROOT}/results/stats", f"{ROOT}/report_tables"
J = lambda p: json.load(open(f"{ST}/{p}"))
esc = lambda s: str(s).replace("_", r"\_").replace("%", r"\%")
f3 = lambda x: "--" if x is None else f"{x:.3f}"


def table(name, caption, label, header, rows, align=None, best_col=None):
    """Write one booktabs table; bold the best value in `best_col` (index into the row)."""
    align = align or "l" + "r" * (len(header) - 1)
    if best_col is not None:
        vals = [r[best_col] for r in rows if isinstance(r[best_col], float)]
        top = max(vals) if vals else None
    lines = [r"\begin{table}[t]", r"\centering\small", rf"\begin{{tabular}}{{{align}}}", r"\toprule",
             " & ".join(header) + r" \\", r"\midrule"]
    for r in rows:
        cells = []
        for i, c in enumerate(r):
            s = f3(c) if isinstance(c, float) else esc(c)
            if best_col is not None and i == best_col and isinstance(c, float) and abs(c - top) < 1e-9:
                s = rf"\textbf{{{s}}}"
            cells.append(s)
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", rf"\caption{{{caption}}}", rf"\label{{{label}}}", r"\end{table}"]
    open(f"{OUT}/{name}.tex", "w").write("\n".join(lines) + "\n")
    print("wrote", name)


# ---------------- Phase 1: pretraining summary + local decoding study
rows = []
for r in csv.DictReader(open(f"{ST}/training_summary.csv")):
    rows.append((r["run"], r["total params (M)"], r["active params (M)"], r["train hours"],
                 float(r["supre bpw"]), float(r["high_ bpw"]), float(r["tribu bpw"]), float(r["acts bpw"])))
table("phase1_pretraining", "Phase 1 pretraining (700M tokens each): validation bits per word by source.",
      "tab:p1-pretrain", ["Run", "Params (M)", "Active (M)", "Hours", "SC", "HC", "Trib.", "Acts"], rows)
dec = {**J("decoding_m1_v2.json"), **J("decoding_m2_v2.json"), **J("decoding_m3_v2.json"), **J("decoding_m4_v2.json")}
pick = ["copy_last_128 (baseline)", "greedy_rep1.2_copy0.0", "greedy_rep1.2_copy1.0", "mbr9_copy1.0",
        "m2_mbr9_copy1.0", "m3_mbr9_copy1.0", "m4_greedy_rep1.2_copy0.0", "m4_greedy_promptcopy_gate+1",
        "m4_mbr9_promptcopy_gate+1"]
rows = [(k if k.startswith("m") else "m1 " + k, dec[k]["bertscore_f1"], dec[k].get("bertscore_f1[SC]"),
         dec[k]["rougeL_f1"], dec[k].get("rep3", 0.0)) for k in pick if k in dec]
table("phase1_decoding", "Phase 1 decoding study on 350 validation prompts (local BERTScore, relative comparison only).",
      "tab:p1-decoding", ["Setting", "BERTScore", "SC", "ROUGE-L", "rep3"], rows, best_col=1)

# ---------------- Phase 2: technique batch (strict micro-F1 on dev, first batch)
p2 = J("p2_results.json")
rows = [(k, v.get("backbone", "m1"), v.get("method", "full"), f"{v['trainable']/1e6:.2f}" if "trainable" in v else "--",
         v["best_dev_f1"]) for k, v in p2.items()]
table("phase2_methods", "Phase 2 (L-NER): adaptation methods and backbones, dev strict micro-F1.", "tab:p2",
      ["Run", "Backbone", "Method", "Trainable (M)", "Dev F1"], rows, best_col=4)

# ---------------- Phase 3
p3 = J("p3_submissions.json")
rows = sorted(((k, v["dev_macro"], v["dev_CL"], v["dev_IT"]) for k, v in p3.items()), key=lambda r: -r[1])
table("phase3", "Phase 3 (rhetorical roles): dev macro-F1 overall and per domain.", "tab:p3",
      ["System", "Macro-F1", "CL", "IT"], rows, best_col=1)

# ---------------- Phase 4
p4 = J("p4_submissions.json")
rows = sorted(((k, v["dev_f1"], v["t"]) for k, v in p4.items()), key=lambda r: -r[1])
table("phase4", "Phase 4 (judgment prediction): clean-dev macro-F1 at the dev-tuned threshold.", "tab:p4",
      ["System", "Macro-F1", "Threshold"], rows, best_col=1)

# ---------------- Phase 5
p5 = J("p5_systems.json")
rows = sorted(((k, v["rule"], v["dev_insample"], v["dev_honest"]) for k, v in p5.items()), key=lambda r: -r[3])
table("phase5", "Phase 5 (statute identification): dev macro-F1, in-sample vs honest (cross-fitted).", "tab:p5",
      ["System", "Rule", "In-sample", "Honest"], rows, best_col=3)

# ---------------- Phase 6 (calibrated runs)
rows = []
for f in sorted(glob.glob(f"{ST}/p6_runs/*__cal.json")):
    r = json.load(open(f)); v = r["val_rouge"]
    rows.append((os.path.basename(f)[:-5], f"{r['trainable_M']:.2f}", v["rouge1"], v["rouge2"], v["rougeL"],
                 v["len_ratio"], r["gen"]["eos_bias"]))
rows.sort(key=lambda r: -r[2])
table("phase6", "Phase 6 (summarisation): length-calibrated runs on 100 validation documents.", "tab:p6",
      ["Run", "Trainable (M)", "R-1", "R-2", "R-L", "Length", "eos bias"], rows, best_col=2)

# ---------------- Phase 7
v2 = J("p7_v2.json")["per_run"]
rows = []
for f in sorted(glob.glob(f"{ST}/p7_runs/*.json")):
    r = json.load(open(f)); n = r["name"]
    rows.append((n, r["method"], f"{r['trainable_M']:.3f}", r["dev_dense_only"]["mean"]["F1"],
                 r["dev_dense_only"]["maxsim"]["F1"], r["dev_best"]["F1"], v2.get(n, {}).get("dev_F1_v2")))
rows.sort(key=lambda r: -r[4])
table("phase7", "Phase 7 (prior case retrieval): Dev Micro-F1@7. Dense-only (mean / MaxSim) and fused with "
      "letters-only BM25 (v1) or 1--3-gram BM25 (v2). BM25 alone: 0.394 (letters), 0.445 (1--3-gram).", "tab:p7",
      ["Run", "Method", "Trainable (M)", "Mean", "MaxSim", "Fused v1", "Fused v2"], rows, best_col=4)
