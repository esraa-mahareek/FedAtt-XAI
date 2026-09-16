"""Centralized upper reference (Section 4.6.1)."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fedattxai.data.loaders import build_pooled_data  # noqa: E402
from fedattxai.fl.centralized import train_centralized  # noqa: E402
from fedattxai.pipeline import get_partition, run_dir  # noqa: E402
from fedattxai.utils.common import (export_environment, get_device, load_config,  # noqa: E402
                                    parse_overrides, set_seed)

ap = argparse.ArgumentParser()
ap.add_argument("--config", required=True)
ap.add_argument("--seed", type=int, required=True)
ap.add_argument("--set", nargs="*", default=[])
a = ap.parse_args()
cfg = load_config(a.config, overrides=parse_overrides(a.set))
out = run_dir(cfg, "centralized", a.seed)
export_environment(Path(out) / "environment.json")
set_seed(a.seed)
part = get_partition(cfg, a.seed)
pooled = build_pooled_data(part, cfg, a.seed)
print(train_centralized(pooled, cfg, a.seed, out, get_device())["pooled"])
