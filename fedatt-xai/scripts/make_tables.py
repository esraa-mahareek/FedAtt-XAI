"""Regenerate the result tables from the per-seed JSON outputs.

python scripts/make_tables.py --results /content/results --beta 0.5
Writes CSV files to <results>/tables/.
"""
import argparse
import glob
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fedattxai.evaluation.statistics import paired_comparison  # noqa: E402

TASKS = {"breakhis": 8, "bach": 4, "inbreast": 5, "cbis_ddsm": 8}
METHODS = ["fedavg", "fedprox", "fedbn", "moon", "fedas", "fedatt"]
LABEL = {"fedavg": "FedAvg", "fedprox": "FedProx", "fedbn": "FedBN (LN-adapted)", "moon": "MOON",
         "fedas": "FedAS", "fedatt": "FedAtt"}
COLS = ["accuracy", "macro_precision", "macro_recall", "macro_f1", "auroc", "pr_auc", "ece", "brier"]

ap = argparse.ArgumentParser()
ap.add_argument("--results", required=True)
ap.add_argument("--beta", default="0.5")
a = ap.parse_args()
R = Path(a.results)
T = R / "tables"
T.mkdir(parents=True, exist_ok=True)


def load(task, method, beta=a.beta, N=None, key="global", fname="metrics.json"):
    N = N or TASKS[task]
    out = {}
    for f in sorted(glob.glob(str(R / task / f"beta{beta}_N{N}" / method / "seed*" / fname))):
        seed = int(Path(f).parent.name[4:])
        d = json.load(open(f))
        out[seed] = d[key]["pooled"] if key else d
    return out


def fmt(vals, digits=1):
    v = np.asarray(vals, float)
    sd = v.std(ddof=1) if len(v) > 1 else 0.0
    return f"{v.mean():.{digits}f} ± {sd:.{digits}f}"


# ---------------------------------------------------------------- Tables 9–13
for task in TASKS:
    rows, op_rows = [], []
    variants = [(m, "global", LABEL[m]) for m in METHODS] + \
               [("fedatt_xai", "global", "FedAtt-XAI (global)"), ("fedatt_xai", "personalized", "FedAtt-XAI (personalized)")]
    for m, key, lab in variants:
        res = load(task, m, key=key)
        if not res:
            continue
        rows.append({"Method": lab, "seeds": len(res),
                     **{c: fmt([r[c] for r in res.values()], 3 if c in ("ece", "brier") else 1) for c in COLS}})
        if "threshold" in next(iter(res.values())):
            op_rows.append({"Method": lab, "Threshold": fmt([r["threshold"] for r in res.values()], 2),
                            "Sensitivity": fmt([r["sensitivity"] for r in res.values()]),
                            "Specificity": fmt([r["specificity"] for r in res.values()])})
    pd.DataFrame(rows).to_csv(T / f"table_results_{task}_beta{a.beta}.csv", index=False)
    if op_rows:
        pd.DataFrame(op_rows).to_csv(T / f"table13_operating_point_{task}.csv", index=False)

# ---------------------------------------------------------------- Table 14 (β 0.5 vs 0.1)
rows = []
for m, key, lab in [(m, "global", LABEL[m]) for m in METHODS] + \
        [("fedatt_xai", "global", "FedAtt-XAI (global)"), ("fedatt_xai", "personalized", "FedAtt-XAI (personalized)")]:
    row = {"Method": lab}
    for task in TASKS:
        v05 = [r["macro_f1"] for r in load(task, m, "0.5", key=key).values()]
        v01 = [r["macro_f1"] for r in load(task, m, "0.1", key=key).values()]
        row[task] = f"{np.mean(v05):.1f} / {np.mean(v01):.1f}" if v05 and v01 else ""
    rows.append(row)
pd.DataFrame(rows).to_csv(T / "table14_heterogeneity.csv", index=False)

# ---------------------------------------------------------------- Table 15 (centralized)
rows = []
for task in TASKS:
    cen = {}
    for f in glob.glob(str(R / task / f"beta{a.beta}_N{TASKS[task]}" / "centralized" / "seed*" / "metrics_centralized.json")):
        cen[int(Path(f).parent.name[4:])] = json.load(open(f))["pooled"]
    if not cen:
        continue
    ref = np.mean([r["macro_f1"] for r in cen.values()])
    rows.append({"Task": task, "Model": "Centralized", "Macro-F1": fmt([r["macro_f1"] for r in cen.values()]), "Δ": "—"})
    for m, key, lab in [("fedatt_xai", "global", "FedAtt-XAI (global)"), ("fedatt_xai", "personalized", "FedAtt-XAI (personalized)")]:
        res = load(task, m, key=key)
        if res:
            rows.append({"Task": task, "Model": lab, "Macro-F1": fmt([r["macro_f1"] for r in res.values()]),
                         "Δ": f"{np.mean([r['macro_f1'] for r in res.values()]) - ref:+.1f}"})
pd.DataFrame(rows).to_csv(T / "table15_centralized.csv", index=False)

# ---------------------------------------------------------------- Table 16 (paired, BCa)
rows = []
for task in TASKS:
    prop = load(task, "fedatt_xai", key="personalized")
    for comp in ("fedas", "fedatt"):
        base = load(task, comp)
        seeds = sorted(set(prop) & set(base))
        if len(seeds) < 2:
            continue
        pc = paired_comparison([prop[s]["macro_f1"] for s in seeds], [base[s]["macro_f1"] for s in seeds])
        rows.append({"Task": task, "Comparator": LABEL[comp], "Δ Macro-F1 (pp)": f"{pc['mean_diff']:+.1f}",
                     "95% BCa CI": f"[{pc['ci_low']:.1f}, {pc['ci_high']:.1f}]",
                     "Rank-biserial": f"{pc['rank_biserial']:+.2f}", "seeds": len(seeds)})
pd.DataFrame(rows).to_csv(T / "table16_paired.csv", index=False)

# ---------------------------------------------------------------- Table 21 (τ sweep)
rows = defaultdict(dict)
for task in TASKS:
    per_tau = defaultdict(list)
    for f in glob.glob(str(R / task / f"beta{a.beta}_N{TASKS[task]}" / "fedatt_xai" / "seed*" / "tau_sweep.json")):
        for tau, d in json.load(open(f)).items():
            per_tau[tau].append((d["macro_f1"], d["head_only_pct"]))
    for tau, v in per_tau.items():
        rows[tau][task] = f"{fmt([x for x, _ in v])} ({np.mean([h for _, h in v]):.0f}%)"
pd.DataFrame([{"tau": k, **v} for k, v in rows.items()]).to_csv(T / "table21_tau.csv", index=False)

# ---------------------------------------------------------------- Table 22 (DLG)
rows = []
for m in METHODS + ["fedatt_xai"]:
    row = {"Method": LABEL.get(m, "FedAtt-XAI (federated stage)")}
    for task in TASKS:
        v = [json.load(open(f))["psnr_mean"] for f in
             glob.glob(str(R / task / f"beta{a.beta}_N{TASKS[task]}" / m / "seed*" / "dlg.json"))]
        row[task] = fmt(v) if v else ""
    rows.append(row)
pd.DataFrame(rows).to_csv(T / "table22_dlg.csv", index=False)

# ---------------------------------------------------------------- Tables 23–28 (XAI summaries)
rows = []
for task in list(TASKS) + ["brecahad"]:
    N = TASKS.get(task, 8)
    files = glob.glob(str(R / task / f"beta{a.beta}_N{N}" / "explanations" / "seed*" / "summary.json"))
    per_key = defaultdict(list)
    for f in files:
        for k, v in json.load(open(f)).items():
            per_key[k].append(v["mean"])
    for k, v in sorted(per_key.items()):
        rows.append({"task": task, "metric": k, "value": fmt(v, 2), "seeds": len(v)})
pd.DataFrame(rows).to_csv(T / "tables23_28_explanations_long.csv", index=False)

# ---------------------------------------------------------------- cost (Section 4.11)
rows = []
for task in TASKS:
    for m in METHODS + ["fedatt_xai"]:
        costs = [json.load(open(f))["cost"] for f in
                 glob.glob(str(R / task / f"beta{a.beta}_N{TASKS[task]}" / m / "seed*" / "metrics.json"))]
        if costs:
            rows.append({"task": task, "method": m, "wall_clock_h": fmt([c["wall_clock_h"] for c in costs], 2),
                         "peak_mem_gb": fmt([c["peak_mem_gb"] or 0 for c in costs], 2)})
pd.DataFrame(rows).to_csv(T / "cost.csv", index=False)
print("tables written to", T)
