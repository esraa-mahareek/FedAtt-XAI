"""Train and evaluate one federated method on one task and seed.

python scripts/train_federated.py --config configs/breakhis.yaml --method fedatt_xai --seed 0
methods: fedavg fedprox fedbn moon fedas fedatt fedatt_xai fedas_xai
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fedattxai.fl.server import FederatedRun  # noqa: E402
from fedattxai.models.backbones import count_parameters  # noqa: E402
from fedattxai.pipeline import get_clients, run_dir  # noqa: E402
from fedattxai.utils.common import (export_environment, get_device, load_config,  # noqa: E402
                                    parse_overrides, save_json, set_seed)

METHODS = ["fedavg", "fedprox", "fedbn", "moon", "fedas", "fedatt", "fedatt_xai", "fedas_xai"]

ap = argparse.ArgumentParser()
ap.add_argument("--config", required=True)
ap.add_argument("--method", required=True, choices=METHODS)
ap.add_argument("--seed", type=int, required=True)
ap.add_argument("--tag", default="", help="suffix for ablation variants, e.g. resnet50")
ap.add_argument("--set", nargs="*", default=[], help="dotted overrides, e.g. fl.rounds=5")
a = ap.parse_args()

cfg = load_config(a.config, overrides=parse_overrides(a.set))
out = run_dir(cfg, a.method, a.seed, a.tag)
out.mkdir(parents=True, exist_ok=True)
save_json(cfg, out / "config_resolved.json")
export_environment(out / "environment.json")
set_seed(a.seed)
dev = get_device()

part, clients = get_clients(cfg, a.seed)
run = FederatedRun(a.method, clients, cfg, a.seed, out, dev)
print("parameters:", count_parameters(run.model))
cost = run.train()

results = {"method": a.method, "task": cfg["task"], "seed": a.seed, "cost": cost,
           "client_sizes": {c.cid: {s: c.n(s) for s in ("train", "val", "test")} for c in clients},
           "global": run.evaluate_global()}
if a.method.endswith("_xai"):
    results["personalized"] = run.personalize_and_evaluate()
save_json(results, out / "metrics.json")
print("global macro-F1:", round(results["global"]["pooled"]["macro_f1"], 2))
if "personalized" in results:
    print("personalized macro-F1:", round(results["personalized"]["pooled"]["macro_f1"], 2))
