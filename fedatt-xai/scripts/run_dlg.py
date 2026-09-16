"""Gradient-inversion stress test (Section 4.9, Table 22).
Attacks the round-10 global model of a method with the gradient of one client
batch of 32 training images; 10 attacked batches per seed."""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fedattxai.models.backbones import create_model  # noqa: E402
from fedattxai.pipeline import get_clients, run_dir  # noqa: E402
from fedattxai.privacy.dlg import dlg_attack  # noqa: E402
from fedattxai.utils.common import get_device, load_config, parse_overrides, save_json, set_seed  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--config", required=True)
ap.add_argument("--method", required=True)
ap.add_argument("--seed", type=int, required=True)
ap.add_argument("--set", nargs="*", default=[])
a = ap.parse_args()
cfg = load_config(a.config, overrides=parse_overrides(a.set))
pc = cfg["privacy"]
dev = get_device()
set_seed(a.seed)
src = run_dir(cfg, a.method, a.seed)
part, clients = get_clients(cfg, a.seed)
ck = torch.load(src / f"round{pc['round']}.pt")
model = create_model(cfg["model"]["backbone"], len(cfg["label_space"]), pretrained=False).to(dev)

rng = np.random.default_rng(a.seed)
results = []
for b in range(pc["attacked_batches"]):
    c = clients[int(rng.integers(len(clients)))]
    state = dict(ck["global"])
    if ck["local"]:
        state.update(ck["local"][c.cid])
    model.load_state_dict(state)
    ds = c.dataset("train", augment=False)
    idx = rng.choice(len(ds), size=min(pc["batch_size"], len(ds)), replace=False)
    x = torch.stack([ds[i][0] for i in idx]).to(dev)
    y = torch.tensor([ds[i][1] for i in idx]).to(dev)
    r = dlg_attack(model, x, y, ds.normalize, ds.denormalize, pc["iterations"], seed=a.seed * 100 + b)
    r["client"] = c.cid
    results.append(r)
    print(f"batch {b}: PSNR {r['psnr_mean']:.2f} dB")
save_json({"psnr_mean": float(np.mean([r["psnr_mean"] for r in results])), "batches": results},
          src / "dlg.json")
