"""Seeding, configuration, logging and environment export (Section 4.2)."""
from __future__ import annotations

import copy
import hashlib
import json
import os
import platform
import random
import time
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
import yaml


# --------------------------------------------------------------------------- #
# Reproducibility
# --------------------------------------------------------------------------- #
def set_seed(seed: int, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        # warn_only=True: unavoidable nondeterministic kernels are logged, not fatal
        torch.use_deterministic_algorithms(True, warn_only=True)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
def _deep_update(base: dict, upd: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (upd or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_update(out[k], v)
        else:
            out[k] = v
    return out


def load_config(task_cfg: str, base_cfg: str = "configs/base.yaml",
                overrides: Dict[str, Any] | None = None) -> dict:
    """base.yaml <- task yaml <- dotted CLI overrides (e.g. fl.rounds=5)."""
    with open(base_cfg) as f:
        cfg = yaml.safe_load(f)
    with open(task_cfg) as f:
        cfg = _deep_update(cfg, yaml.safe_load(f))
    for dotted, value in (overrides or {}).items():
        node = cfg
        keys = dotted.split(".")
        for k in keys[:-1]:
            node = node.setdefault(k, {})
        node[keys[-1]] = value
    return cfg


def parse_overrides(pairs) -> Dict[str, Any]:
    out = {}
    for p in pairs or []:
        k, v = p.split("=", 1)
        out[k] = yaml.safe_load(v)
    return out


# --------------------------------------------------------------------------- #
# IO / logging
# --------------------------------------------------------------------------- #
def _json_default(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(type(o))


def save_json(obj: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=_json_default)


def load_json(path: str | Path) -> Any:
    with open(path) as f:
        return json.load(f)


def export_environment(path: str | Path) -> dict:
    """Export package versions and detected hardware at the start of every run."""
    info = {
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "torch": torch.__version__,
        "cuda_build": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "gpu_mem_gb": (torch.cuda.get_device_properties(0).total_memory / 1e9
                       if torch.cuda.is_available() else None),
    }
    try:
        import psutil
        info["ram_gb"] = psutil.virtual_memory().total / 1e9
    except ImportError:
        info["ram_gb"] = None
    for mod in ["timm", "captum", "sklearn", "scipy", "numpy", "pandas", "matplotlib"]:
        try:
            info[mod] = __import__(mod).__version__
        except Exception:  # noqa: BLE001
            info[mod] = None
    save_json(info, path)
    return info


class JsonlLogger:
    """Per-round training log (one JSON record per line)."""

    def __init__(self, path: str | Path, verbose: bool = True):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.verbose = verbose

    def log(self, **record):
        record["time"] = time.time()
        with open(self.path, "a") as f:
            f.write(json.dumps(record, default=_json_default) + "\n")
        if self.verbose:
            print({k: (round(v, 4) if isinstance(v, float) else v)
                   for k, v in record.items() if k != "time"}, flush=True)


def sha256_state_dict(state: dict) -> str:
    """Checkpoint hash released with the implementation package."""
    h = hashlib.sha256()
    for k in sorted(state):
        h.update(k.encode())
        h.update(state[k].detach().cpu().float().contiguous().numpy().tobytes())
    return h.hexdigest()
