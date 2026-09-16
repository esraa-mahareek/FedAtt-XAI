"""Dataset index builders (Sections 3.1.3, 4.1, 4.1.1).

Every builder returns a pandas DataFrame with one row per image:

    image_id   unique string id
    path       absolute path of the image file (PNG / TIFF / DICOM)
    label      integer class index in the task label space
    group      grouping unit (patient / case / image) used for partitioning
    modality   "histo" or "mammo"
    ann        (optional) annotation reference used by the localization analysis

Expected raw directory layouts are documented in README.md.
"""
from __future__ import annotations

import glob
import os
import re
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

LABEL_SPACES = {
    "breakhis": ["benign", "malignant"],
    "bach": ["Normal", "Benign", "InSitu", "Invasive"],
    "inbreast": ["negative", "positive"],
    "cbis_ddsm": ["benign", "malignant"],
}
MODALITY = {"breakhis": "histo", "bach": "histo", "inbreast": "mammo",
            "cbis_ddsm": "mammo", "brecahad": "histo"}


# --------------------------------------------------------------------------- #
# BreaKHis — patient id is encoded in the filename
#   SOB_B_TA-14-4659-40-001.png  ->  class B, patient 14-4659, magnification 40
# --------------------------------------------------------------------------- #
_BREAKHIS_RE = re.compile(r"SOB_(?P<cls>[BM])_(?P<sub>[A-Z]+)-(?P<pid>\d+-\d+[A-Z]*)-(?P<mag>\d+)-(?P<idx>\d+)")


def parse_breakhis_name(fname: str) -> dict:
    m = _BREAKHIS_RE.search(os.path.basename(fname))
    if m is None:
        raise ValueError(f"Unrecognised BreaKHis filename: {fname}")
    return m.groupdict()


def build_breakhis(root: str) -> pd.DataFrame:
    files = sorted(glob.glob(os.path.join(root, "**", "SOB_*.png"), recursive=True))
    if not files:
        raise FileNotFoundError(f"No BreaKHis PNGs under {root}")
    rows = []
    for f in files:
        meta = parse_breakhis_name(f)
        rows.append(dict(
            image_id=Path(f).stem, path=f,
            label=0 if meta["cls"] == "B" else 1,
            group=f"patient_{meta['pid']}",
            magnification=int(meta["mag"]), subtype=meta["sub"],
            modality="histo", ann=None))
    df = pd.DataFrame(rows).drop_duplicates("image_id").reset_index(drop=True)
    return df


# --------------------------------------------------------------------------- #
# BACH — no patient identifiers: each image is its own group
# --------------------------------------------------------------------------- #
def build_bach(root: str) -> pd.DataFrame:
    photos = os.path.join(root, "Photos") if os.path.isdir(os.path.join(root, "Photos")) else root
    rows = []
    for ci, cname in enumerate(LABEL_SPACES["bach"]):
        for f in sorted(glob.glob(os.path.join(photos, cname, "*.tif"))):
            iid = f"{cname}_{Path(f).stem}"
            rows.append(dict(image_id=iid, path=f, label=ci, group=f"image_{iid}",
                             modality="histo", ann=None))
    if not rows:
        raise FileNotFoundError(f"No BACH images under {photos}")
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# INbreast — BI-RADS 1–2 -> negative, 4–6 -> positive, BI-RADS 3 excluded.
# Case id = second token of the DICOM filename
#   20586908_6c613a14b80a8591_MG_R_CC_ANON.dcm -> case 6c613a14b80a8591
# --------------------------------------------------------------------------- #
def _birads_to_int(v) -> int | None:
    m = re.match(r"\s*(\d)", str(v))
    return int(m.group(1)) if m else None


def build_inbreast(root: str, meta_file: str | None = None) -> pd.DataFrame:
    dicoms = {Path(f).name.split("_")[0]: f
              for f in glob.glob(os.path.join(root, "AllDICOMs", "*.dcm"))}
    if meta_file is None:
        cands = glob.glob(os.path.join(root, "INbreast.xls*")) + glob.glob(os.path.join(root, "INbreast.csv"))
        if not cands:
            raise FileNotFoundError("INbreast.xls / INbreast.csv not found")
        meta_file = cands[0]
    meta = (pd.read_csv(meta_file, sep=None, engine="python") if meta_file.endswith(".csv")
            else pd.read_excel(meta_file))
    meta.columns = [c.strip() for c in meta.columns]
    fcol = next(c for c in meta.columns if c.lower().startswith("file name"))
    bcol = next(c for c in meta.columns if c.lower().replace("-", "").startswith("birads"))
    rows, excluded = [], 0
    for _, r in meta.iterrows():
        if pd.isna(r[fcol]):
            continue
        fid = str(int(r[fcol])) if not isinstance(r[fcol], str) else r[fcol].strip()
        if fid not in dicoms:
            continue
        b = _birads_to_int(r[bcol])
        if b is None:
            continue
        if b == 3:
            excluded += 1
            continue
        label = 0 if b in (1, 2) else 1
        path = dicoms[fid]
        case = Path(path).name.split("_")[1]
        xml = os.path.join(root, "AllXML", f"{fid}.xml")
        rows.append(dict(image_id=fid, path=path, label=label, group=f"case_{case}",
                         birads=b, modality="mammo",
                         ann=xml if os.path.exists(xml) else None))
    df = pd.DataFrame(rows)
    df.attrs["excluded_birads3"] = excluded
    return df


# --------------------------------------------------------------------------- #
# CBIS-DDSM — regrouped by patient; official split NOT used.
# One row per full mammogram (patient, side, view). BENIGN and
# BENIGN_WITHOUT_CALLBACK -> benign; malignant if any abnormality is malignant.
# --------------------------------------------------------------------------- #
def _resolve_tcia_path(dicom_root: str, rel: str) -> str | None:
    """CBIS-DDSM csv paths look like
    'Mass-Training_P_00001_LEFT_CC/1.3.6.../1.3.6.../000000.dcm' while the
    TCIA download stores the series under differently named UID folders.
    Resolve by the top-level folder and take the first DICOM found."""
    rel = str(rel).strip().replace("\n", "")
    top = rel.split("/")[0]
    hits = sorted(glob.glob(os.path.join(dicom_root, top, "**", "*.dcm"), recursive=True))
    return hits[0] if hits else None


def _resolve_roi_mask(dicom_root: str, rel: str) -> str | None:
    """ROI folders contain a cropped image and a mask; the mask is the
    larger file (full-mammogram size)."""
    rel = str(rel).strip().replace("\n", "")
    top = rel.split("/")[0]
    hits = glob.glob(os.path.join(dicom_root, top, "**", "*.dcm"), recursive=True)
    if not hits:
        return None
    return max(hits, key=os.path.getsize)


def build_cbis_ddsm(root: str, dicom_root: str | None = None) -> pd.DataFrame:
    dicom_root = dicom_root or os.path.join(root, "CBIS-DDSM")
    csvs = sorted(glob.glob(os.path.join(root, "*case_description*.csv")))
    if not csvs:
        raise FileNotFoundError("CBIS-DDSM *_case_description_*.csv files not found")
    ab = pd.concat([pd.read_csv(c) for c in csvs], ignore_index=True)
    ab.columns = [c.strip().lower() for c in ab.columns]
    side_col = "left or right breast"
    ab["malignant"] = ab["pathology"].str.upper().eq("MALIGNANT")
    rows = []
    for (pid, side, view), g in ab.groupby(["patient_id", side_col, "image view"]):
        label = int(g["malignant"].any())
        img = _resolve_tcia_path(dicom_root, g["image file path"].iloc[0])
        if img is None:
            continue
        # reference mask = union of ROIs whose pathology determines the image label
        det = g[g["malignant"]] if label == 1 else g
        masks = [m for m in (_resolve_roi_mask(dicom_root, p) for p in det["roi mask file path"]) if m]
        rows.append(dict(image_id=f"{pid}_{side}_{view}", path=img, label=label,
                         group=f"patient_{pid}", modality="mammo",
                         ann="|".join(masks) if masks else None))
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# BreCaHAD — explainability only (no classification task).
# groundTruth/<name>.json holds point annotations with normalised x,y for
# mitosis, apoptosis, tumor, non_tumor, lumen, non_lumen.
# --------------------------------------------------------------------------- #
def build_brecahad(root: str) -> pd.DataFrame:
    imgs = sorted(glob.glob(os.path.join(root, "images", "*.tif")))
    rows = []
    for f in imgs:
        gt = os.path.join(root, "groundTruth", Path(f).stem + ".json")
        rows.append(dict(image_id=Path(f).stem, path=f, label=-1,
                         group=f"image_{Path(f).stem}", modality="histo",
                         ann=gt if os.path.exists(gt) else None))
    return pd.DataFrame(rows)


BUILDERS = {
    "breakhis": build_breakhis, "bach": build_bach, "inbreast": build_inbreast,
    "cbis_ddsm": build_cbis_ddsm, "brecahad": build_brecahad,
}


def build_index(task: str, root: str, **kw) -> pd.DataFrame:
    df = BUILDERS[task](root, **kw)
    df["task"] = task
    return df


def summarize_index(df: pd.DataFrame) -> dict:
    return {
        "images": int(len(df)),
        "groups": int(df["group"].nunique()),
        "per_class_images": df["label"].value_counts().sort_index().to_dict(),
        "per_class_groups": df.groupby("label")["group"].nunique().to_dict(),
    }
