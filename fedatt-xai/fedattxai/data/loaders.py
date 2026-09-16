"""Per-client data stores, augmentation and DataLoaders.

The deterministic preprocessing of every client (fitted on its own training
partition) is applied once and cached as uint8 arrays under
<cache>/<task>/beta<β>_N<N>/seed<s>/client<i>/, so images are not re-read
every round. Augmentation is applied on the fly to the training split only.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
import torchvision.transforms.v2 as T
from torch.utils.data import DataLoader, Dataset

from .preprocessing import ClientPreprocessor


# --------------------------------------------------------------------------- #
# Augmentation (identical for every compared method, Section 3.3)
# --------------------------------------------------------------------------- #
class MammoBrightnessContrast(torch.nn.Module):
    """Random brightness and contrast scaling in [0.90, 1.10], applied
    identically to the three replicated channels."""

    def __init__(self, lo=0.9, hi=1.1):
        super().__init__()
        self.lo, self.hi = lo, hi

    def forward(self, x):  # x float in [0,1], 3xHxW
        b = torch.empty(1).uniform_(self.lo, self.hi)
        c = torch.empty(1).uniform_(self.lo, self.hi)
        m = x.mean()
        return ((x - m) * c + m * b).clamp(0, 1)


def train_augmentation(modality: str):
    if modality == "histo":
        return T.Compose([
            T.RandomHorizontalFlip(), T.RandomVerticalFlip(),
            T.RandomRotation(15, interpolation=T.InterpolationMode.BILINEAR),
            T.ColorJitter(brightness=0.10, contrast=0.10, saturation=0.10),
        ])
    # mammography: no vertical flip (CC / MLO orientation carries meaning)
    return T.Compose([
        T.RandomHorizontalFlip(),
        T.RandomRotation(15, interpolation=T.InterpolationMode.BILINEAR),
        MammoBrightnessContrast(0.9, 1.1),
    ])


class ArrayDataset(Dataset):
    def __init__(self, images: np.ndarray, labels: np.ndarray, pp: ClientPreprocessor,
                 augment=None, ids: List[str] | None = None):
        self.images, self.labels, self.pp, self.augment = images, labels, pp, augment
        self.ids = ids
        self.mean = torch.tensor(pp.mean).view(3, 1, 1)
        self.std = torch.tensor(pp.std).view(3, 1, 1)

    def __len__(self):
        return len(self.labels)

    def to_unit(self, i) -> torch.Tensor:
        return torch.from_numpy(np.array(self.images[i])).permute(2, 0, 1).float() / 255.0

    def normalize(self, x):
        return (x - self.mean) / self.std

    def denormalize(self, x):
        return x * self.std.to(x.device) + self.mean.to(x.device)

    def __getitem__(self, i):
        x = self.to_unit(i)
        if self.augment is not None:
            x = self.augment(x)
        return self.normalize(x), int(self.labels[i])


# --------------------------------------------------------------------------- #
# Client store
# --------------------------------------------------------------------------- #
@dataclass
class ClientData:
    cid: int
    pp: ClientPreprocessor
    frames: Dict[str, pd.DataFrame]
    arrays: Dict[str, np.ndarray]

    def n(self, split="train"):
        return len(self.frames[split])

    def dataset(self, split, augment=False) -> ArrayDataset:
        aug = train_augmentation(self.pp.modality) if (augment and split == "train") else None
        return ArrayDataset(self.arrays[split], self.frames[split]["label"].to_numpy(),
                            self.pp, aug, self.frames[split]["image_id"].tolist())

    def loader(self, split, batch_size, shuffle=None, augment=None, seed=0, workers=2):
        shuffle = (split == "train") if shuffle is None else shuffle
        augment = (split == "train") if augment is None else augment
        g = torch.Generator().manual_seed(seed)
        return DataLoader(self.dataset(split, augment), batch_size=batch_size, shuffle=shuffle,
                          num_workers=workers, pin_memory=torch.cuda.is_available(),
                          generator=g, drop_last=False, persistent_workers=False)


def build_client_data(part: pd.DataFrame, cid: int, cfg: dict, cache_dir: str | Path,
                      seed: int) -> ClientData:
    """Fit preprocessing on the client's training split, cache processed arrays."""
    cache = Path(cache_dir) / f"client{cid}"
    cache.mkdir(parents=True, exist_ok=True)
    sub = part[part["client"] == cid]
    frames = {s: sub[sub["split"] == s].reset_index(drop=True) for s in ("train", "val", "test")}
    modality = sub["modality"].iloc[0]
    pp_path = cache / "preprocessor.json"
    pcfg = cfg.get("preprocessing", {})
    if pp_path.exists():
        pp = ClientPreprocessor.load(pp_path)
    else:
        rng = np.random.default_rng(seed * 1000 + cid)
        pp = ClientPreprocessor.fit(frames["train"]["path"].tolist(), modality, pcfg, rng)  # train only
        pp.save(pp_path)
    arrays = {}
    for s, fr in frames.items():
        f = cache / f"{s}.npy"
        ids_f = cache / f"{s}_ids.txt"
        if f.exists() and ids_f.exists() and ids_f.read_text().split("\n") == fr["image_id"].tolist():
            arrays[s] = np.load(f, mmap_mode="r")
        else:
            arr = np.stack([pp.to_uint8(p, pcfg) for p in fr["path"]]) if len(fr) else \
                np.zeros((0, 224, 224, 3), np.uint8)
            np.save(f, arr)
            ids_f.write_text("\n".join(fr["image_id"].tolist()))
            arrays[s] = arr
    return ClientData(cid=cid, pp=pp, frames=frames, arrays=arrays)


def build_all_clients(part: pd.DataFrame, cfg: dict, seed: int) -> List[ClientData]:
    cache = Path(cfg["paths"]["cache"]) / cfg["task"] / \
        f"beta{cfg['partition']['beta']}_N{cfg['partition']['num_clients']}" / f"seed{seed}"
    return [build_client_data(part, int(i), cfg, cache, seed)
            for i in sorted(part["client"].unique())]


def build_pooled_data(part: pd.DataFrame, cfg: dict, seed: int) -> ClientData:
    """Centralized reference: preprocessing fitted on the pooled training partition."""
    pooled = part.copy()
    pooled["client"] = 0
    cache = Path(cfg["paths"]["cache"]) / cfg["task"] / \
        f"beta{cfg['partition']['beta']}_N{cfg['partition']['num_clients']}" / f"seed{seed}" / "pooled"
    return build_client_data(pooled, 0, cfg, cache, seed)
