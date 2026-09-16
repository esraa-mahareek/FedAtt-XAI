"""Sensitivity to the freeze threshold τ (Table 21) and the uniform head-only /
full fine-tuning ablation rows (Table 17). Re-uses the saved W_g* of a
fedatt_xai run; only the personalisation stage is recomputed."""
import argparse
import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fedattxai.fl.server import FederatedRun  # noqa: E402
from fedattxai.pipeline import get_clients, run_dir  # noqa: E402
from fedattxai.utils.common import get_device, load_config, parse_overrides, save_json, set_seed  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--config", required=True)
ap.add_argument("--seed", type=int, required=True)
ap.add_argument("--method", default="fedatt_xai")
ap.add_argument("--taus", nargs="*", default=["0", "50", "100", "250", "500", "1000", "2000", "inf"])
ap.add_argument("--set", nargs="*", default=[])
a = ap.parse_args()
cfg = load_config(a.config, overrides=parse_overrides(a.set))
set_seed(a.seed)
src = run_dir(cfg, a.method, a.seed)
part, clients = get_clients(cfg, a.seed)
run = FederatedRun(a.method, clients, cfg, a.seed, src / "tau_sweep", get_device())
ck = torch.load(src / "global_final.pt")
out = {}
for t in a.taus:
    tau = math.inf if t == "inf" else float(t)
    run.global_state, run.local_states = ck["global"], ck["local"]
    set_seed(a.seed)
    r = run.personalize_and_evaluate(tau=tau, tag=f"tau{t}")
    out[t] = {"macro_f1": r["pooled"]["macro_f1"], "head_only_pct": r["head_only_pct"], "pooled": r["pooled"]}
    print(t, round(out[t]["macro_f1"], 2), f"{out[t]['head_only_pct']:.0f}% head-only")
save_json(out, src / "tau_sweep.json")
