"""Phase 0: build leakage-controlled client partitions for every seed and
export split identifiers (released with the package)."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fedattxai.pipeline import get_partition, load_index, partition_path  # noqa: E402
from fedattxai.utils.common import load_config, parse_overrides  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--config", required=True)
ap.add_argument("--seeds", type=int, nargs="*")
ap.add_argument("--set", nargs="*", default=[])
a = ap.parse_args()
cfg = load_config(a.config, overrides=parse_overrides(a.set))
df = load_index(cfg)
for s in (a.seeds if a.seeds is not None else cfg["seeds"]):
    part = get_partition(cfg, s, df)
    print(f"seed {s}: {partition_path(cfg, s)}")
    print(part.groupby(["client", "split"]).size().unstack(fill_value=0))
