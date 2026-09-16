"""Near-duplicate audit of the BACH partitions (Section 4.1.1, Table 6).

Each image -> 64-bit perceptual hash and the CLS embedding of the
ImageNet-21k-pretrained ViT-Small/16 (before any federated training).
A pair is flagged when Hamming distance ≤ 6 bits OR cosine similarity ≥ 0.97.
Flagged pairs are exported with thumbnails for visual confirmation; the
'confirmed' column must be filled in manually.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import imagehash
import numpy as np
import pandas as pd
import torch
from PIL import Image

from ..models.backbones import create_model, features
from .preprocessing import read_rgb


def audit_partition(part: pd.DataFrame, out_dir: str | Path, device, hamming_max: int = 6,
                    cos_min: float = 0.97) -> pd.DataFrame:
    out = Path(out_dir)
    (out / "pairs").mkdir(parents=True, exist_ok=True)
    thumbs = [cv2.resize(read_rgb(p, 1024), (224, 224), interpolation=cv2.INTER_AREA) for p in part["path"]]
    hashes = [imagehash.phash(Image.fromarray(t), hash_size=8) for t in thumbs]

    model = create_model("vit_small", num_classes=0, pretrained=True).to(device).eval()
    mean = torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1)
    embs = []
    with torch.no_grad():
        for i in range(0, len(thumbs), 64):
            x = torch.from_numpy(np.stack(thumbs[i:i + 64])).permute(0, 3, 1, 2).float() / 255
            x = ((x - mean) / 0.5).to(device)
            embs.append(torch.nn.functional.normalize(features(model, x), dim=-1).cpu())
    E = torch.cat(embs)
    cos = (E @ E.T).numpy()

    rows = []
    n = len(part)
    splits = part["split"].tolist()
    ids = part["image_id"].tolist()
    for i in range(n):
        for j in range(i + 1, n):
            ham = hashes[i] - hashes[j]
            hit_h, hit_e = ham <= hamming_max, cos[i, j] >= cos_min
            if hit_h or hit_e:
                pair = {splits[i], splits[j]}
                rows.append(dict(a=ids[i], b=ids[j], split_a=splits[i], split_b=splits[j],
                                 hamming=int(ham), cosine=float(cos[i, j]), flag_hash=hit_h,
                                 flag_embedding=hit_e, cross_train_val=pair == {"train", "val"},
                                 cross_train_test=pair == {"train", "test"}, confirmed=None))
                cv2.imwrite(str(out / "pairs" / f"{ids[i]}__{ids[j]}.png"),
                            cv2.cvtColor(np.hstack([thumbs[i], thumbs[j]]), cv2.COLOR_RGB2BGR))
    df = pd.DataFrame(rows)
    df.to_csv(out / "near_duplicates.csv", index=False)
    return df
