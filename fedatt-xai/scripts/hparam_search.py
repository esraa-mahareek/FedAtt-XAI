"""Random search, 20 trials per method, validation partitions only (Section 4.3).

Objective: sample-weighted federated validation loss of the reported global
model (and, for fedatt_xai's τ, of the personalised models). Test data are
never read. The best configuration is written to <results>/hparams/<task>/<method>.json
and must be passed to the final runs with --set.
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fedattxai.fl.personalization import personalize_client  # noqa: E402
from fedattxai.fl.server import FederatedRun  # noqa: E402
from fedattxai.pipeline import get_clients  # noqa: E402
from fedattxai.utils.common import get_device, load_config, parse_overrides, save_json, set_seed  # noqa: E402

SPACES = {
    "fedprox": lambda r: {"methods.fedprox.mu": float(10 ** r.uniform(-3, 0))},
    "moon": lambda r: {"methods.moon.temperature": float(r.choice([0.1, 0.2, 0.5, 1.0])),
                       "methods.moon.mu": float(10 ** r.uniform(-1, 1))},
    "fedatt": lambda r: {"methods.fedatt.temperature": float(10 ** r.uniform(-2, 2))},
    "fedatt_xai": lambda r: {"methods.fedatt.temperature": float(10 ** r.uniform(-2, 2)),
                             "personalization.tau": int(r.choice([250, 500, 1000]))},
    "fedas": lambda r: {"methods.fedas.align_epochs": int(r.choice([1, 2]))},
}

ap = argparse.ArgumentParser()
ap.add_argument("--config", required=True)
ap.add_argument("--method", required=True, choices=list(SPACES))
ap.add_argument("--set", nargs="*", default=[])
a = ap.parse_args()
base = parse_overrides(a.set)
cfg0 = load_config(a.config, overrides=base)
seed = cfg0["hparam_search"]["seed"]
rng = np.random.default_rng(12345)
dev = get_device()
trials = []
for t in range(cfg0["hparam_search"]["trials"]):
    hp = SPACES[a.method](rng)
    cfg = load_config(a.config, overrides={**base, **hp, "fl.save_rounds": []})
    set_seed(seed)
    part, clients = get_clients(cfg, seed)
    out = Path(cfg["paths"]["results"]) / "hparams" / cfg["task"] / a.method / f"trial{t}"
    run = FederatedRun(a.method, clients, cfg, seed, out, dev)
    run.train()
    obj = run.federated_val_loss(run.global_state)
    if a.method == "fedatt_xai":
        # τ is judged on the personalised models' validation loss only (no test access)
        tot, n = 0.0, 0
        for i, c in enumerate(clients):
            run.model.load_state_dict(run.client_state(i))
            p = personalize_client(run.model, c, cfg, dev, seed)
            best_v = min(h["val_loss"] for h in p["val_history"] if h["epoch"] == p["selected_epochs"])
            tot += best_v * c.n("val")
            n += c.n("val")
        obj = tot / max(n, 1)
    trials.append({"trial": t, "hparams": hp, "val_objective": obj})
    print(t, hp, round(obj, 4))
best = min(trials, key=lambda d: d["val_objective"])
save_json({"best": best, "trials": trials},
          Path(cfg0["paths"]["results"]) / "hparams" / cfg0["task"] / f"{a.method}.json")
print("best:", best)
