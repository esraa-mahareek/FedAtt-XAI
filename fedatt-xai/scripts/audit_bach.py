"""Near-duplicate audit of the realised BACH partitions (Table 6)."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fedattxai.data.audit import audit_partition  # noqa: E402
from fedattxai.pipeline import get_partition, run_dir  # noqa: E402
from fedattxai.utils.common import get_device, load_config, parse_overrides  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--config", default="configs/bach.yaml")
ap.add_argument("--set", nargs="*", default=[])
a = ap.parse_args()
cfg = load_config(a.config, overrides=parse_overrides(a.set))
for s in cfg["seeds"]:
    part = get_partition(cfg, s)
    df = audit_partition(part, run_dir(cfg, "bach_audit", s), get_device())
    if len(df):
        print(f"seed {s}: hash={int(df.flag_hash.sum())} emb={int(df.flag_embedding.sum())} "
              f"train-val={int(df.cross_train_val.sum())} train-test={int(df.cross_train_test.sum())}")
    else:
        print(f"seed {s}: no flagged pairs")
