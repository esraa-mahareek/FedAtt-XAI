"""Modality-specific preprocessing fitted on the client TRAINING partition only
(Section 3.3, Algorithm 1 line 10).

Histopathology : 8-bit RGB -> Macenko stain normalisation to a reference
                 estimated from the client's own training images -> bilinear
                 resize 224 -> per-channel mean/std from the training partition.
Mammography    : DICOM -> rescale slope/intercept -> VOI window (if stored)
                 -> invert MONOCHROME1 -> clip at training 1st/99th percentiles
                 -> min-max [0,1] -> 3 channels -> resize 224.
"""
from __future__ import annotations

import json
import plistlib
from dataclasses import asdict, dataclass, field
from typing import List, Optional

import cv2
import numpy as np

IMG_SIZE = 224


# =========================================================================== #
# Readers
# =========================================================================== #
def read_rgb(path: str, max_side: Optional[int] = None) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:  # some TIFFs need PIL
        from PIL import Image
        img = np.array(Image.open(path).convert("RGB"))[:, :, ::-1]
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if max_side and max(img.shape[:2]) > max_side:
        s = max_side / max(img.shape[:2])
        img = cv2.resize(img, (int(img.shape[1] * s), int(img.shape[0] * s)), interpolation=cv2.INTER_AREA)
    return img


def read_dicom_windowed(path: str) -> np.ndarray:
    """Returns float32 image where larger values = denser tissue."""
    import pydicom
    try:
        from pydicom.pixels import apply_modality_lut, apply_voi_lut
    except ImportError:  # pydicom < 3
        from pydicom.pixel_data_handlers.util import apply_modality_lut, apply_voi_lut

    ds = pydicom.dcmread(path)
    arr = ds.pixel_array
    arr = apply_modality_lut(arr, ds)                      # slope / intercept
    if "WindowCenter" in ds and "WindowWidth" in ds:
        arr = apply_voi_lut(arr, ds)                       # stored window
    arr = arr.astype(np.float32)
    if str(getattr(ds, "PhotometricInterpretation", "")).upper() == "MONOCHROME1":
        arr = arr.max() - arr
    return arr


# =========================================================================== #
# Macenko stain normalisation
# =========================================================================== #
def _od(img: np.ndarray, Io: float = 240.0) -> np.ndarray:
    return -np.log((img.reshape(-1, 3).astype(np.float64) + 1.0) / Io)


def macenko_stain_matrix(img: np.ndarray, Io=240.0, alpha=1.0, beta=0.15):
    """Estimate the 3x2 H&E stain matrix and 99th-percentile concentrations."""
    od = _od(img, Io)
    od_hat = od[np.all(od > beta, axis=1)]
    if len(od_hat) < 50:
        return None, None
    _, eigvecs = np.linalg.eigh(np.cov(od_hat.T))
    plane = eigvecs[:, 1:3]
    proj = od_hat @ plane
    phi = np.arctan2(proj[:, 1], proj[:, 0])
    mn, mx = np.percentile(phi, alpha), np.percentile(phi, 100 - alpha)
    v1 = plane @ np.array([np.cos(mn), np.sin(mn)])
    v2 = plane @ np.array([np.cos(mx), np.sin(mx)])
    HE = np.array([v1, v2]).T if v1[0] > v2[0] else np.array([v2, v1]).T
    HE = HE / np.linalg.norm(HE, axis=0, keepdims=True)
    C = np.linalg.lstsq(HE, od.T, rcond=None)[0]
    maxC = np.percentile(C, 99, axis=1)
    return HE, maxC


@dataclass
class MacenkoReference:
    HE: List[List[float]]
    maxC: List[float]


def fit_macenko_reference(images: List[np.ndarray]) -> MacenkoReference:
    """Reference = median stain vectors / max concentrations over training images."""
    HEs, Cs = [], []
    for im in images:
        HE, maxC = macenko_stain_matrix(im)
        if HE is not None:
            HEs.append(HE)
            Cs.append(maxC)
    if not HEs:  # fall back to the canonical reference of Macenko et al.
        return MacenkoReference(HE=[[0.5626, 0.2159], [0.7201, 0.8012], [0.4062, 0.5581]],
                                maxC=[1.9705, 1.0308])
    HE = np.median(np.stack(HEs), axis=0)
    HE = HE / np.linalg.norm(HE, axis=0, keepdims=True)
    return MacenkoReference(HE=HE.tolist(), maxC=np.median(np.stack(Cs), axis=0).tolist())


def macenko_normalize(img: np.ndarray, ref: MacenkoReference, Io=240.0) -> np.ndarray:
    HE, maxC = macenko_stain_matrix(img)
    if HE is None:
        return img
    od = _od(img, Io)
    C = np.linalg.lstsq(HE, od.T, rcond=None)[0]
    C = C * (np.array(ref.maxC) / (maxC + 1e-8))[:, None]
    out = Io * np.exp(-np.array(ref.HE) @ C)
    return np.clip(out.T.reshape(img.shape), 0, 255).astype(np.uint8)


# =========================================================================== #
# Client-level fitted preprocessing
# =========================================================================== #
@dataclass
class ClientPreprocessor:
    """Parameters estimated on one client's training partition."""
    modality: str
    macenko: Optional[MacenkoReference] = None
    clip_lo: Optional[float] = None
    clip_hi: Optional[float] = None
    mean: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    std: List[float] = field(default_factory=lambda: [1.0, 1.0, 1.0])
    channel_norm: bool = True

    # ------------------------------------------------------------------ fit
    @classmethod
    def fit(cls, paths: List[str], modality: str, cfg: dict, rng: np.random.Generator):
        pp = cls(modality=modality)
        n_fit = cfg.get("fit_max_images", 64)
        sel = list(rng.choice(paths, size=min(n_fit, len(paths)), replace=False))
        if modality == "histo":
            imgs = [read_rgb(p, cfg.get("stain_max_side")) for p in sel]
            pp.macenko = fit_macenko_reference(imgs)
            pp.channel_norm = True
        else:
            vals = []
            for p in sel:
                a = read_dicom_windowed(p)
                sub = a[:: max(1, a.shape[0] // 256), :: max(1, a.shape[1] // 256)]
                vals.append(sub.ravel())
            v = np.concatenate(vals)
            pp.clip_lo, pp.clip_hi = float(np.percentile(v, 1)), float(np.percentile(v, 99))
            pp.channel_norm = cfg.get("mammo_channel_norm", False)
        # channel statistics on the processed 224x224 training images (all of them)
        if pp.channel_norm:
            acc = np.zeros(3)
            acc2 = np.zeros(3)
            n = 0
            for p in paths:
                x = pp.to_uint8(p, cfg).astype(np.float64) / 255.0
                acc += x.reshape(-1, 3).sum(0)
                acc2 += (x.reshape(-1, 3) ** 2).sum(0)
                n += x.shape[0] * x.shape[1]
            mean = acc / n
            pp.mean = mean.tolist()
            pp.std = np.sqrt(np.maximum(acc2 / n - mean ** 2, 1e-8)).tolist()
        return pp

    # ------------------------------------------------------------ transform
    def to_uint8(self, path: str, cfg: dict) -> np.ndarray:
        """Deterministic part of the pipeline -> HxWx3 uint8 at 224x224."""
        if self.modality == "histo":
            img = read_rgb(path, cfg.get("stain_max_side"))
            img = macenko_normalize(img, self.macenko)
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)
        else:
            a = read_dicom_windowed(path)
            a = np.clip(a, self.clip_lo, self.clip_hi)
            a = (a - self.clip_lo) / max(self.clip_hi - self.clip_lo, 1e-6)
            a = cv2.resize(a, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
            img = (np.repeat(a[..., None], 3, axis=2) * 255.0).round().astype(np.uint8)
        return img

    def to_dict(self):
        d = asdict(self)
        return d

    @classmethod
    def from_dict(cls, d):
        d = dict(d)
        if d.get("macenko"):
            d["macenko"] = MacenkoReference(**d["macenko"])
        return cls(**d)

    def save(self, path):
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path):
        with open(path) as f:
            return cls.from_dict(json.load(f))


# =========================================================================== #
# Expert annotations -> binary masks (localization analysis, Section 4.5.2)
# =========================================================================== #
def inbreast_mask(xml_path: str, shape, calc_radius: int = 5) -> np.ndarray:
    """Masses and calcifications from the INbreast OsiriX plist XML."""
    mask = np.zeros(shape, dtype=np.uint8)
    with open(xml_path, "rb") as f:
        pl = plistlib.load(f)
    images = pl.get("Images", [])
    for im in images:
        for roi in im.get("ROIs", []):
            name = str(roi.get("Name", "")).lower()
            if not any(k in name for k in ("mass", "calcification", "cluster", "distortion", "asymmetry")):
                continue
            pts = [tuple(map(float, p.strip("()").split(","))) for p in roi.get("Point_px", [])]
            if not pts:
                continue
            pts = np.array(pts).round().astype(np.int32)
            if len(pts) >= 3:
                cv2.fillPoly(mask, [pts], 1)
            else:
                for x, y in pts:
                    cv2.circle(mask, (int(x), int(y)), calc_radius, 1, -1)
    return mask.astype(bool)


def cbis_mask(mask_paths: str, shape) -> np.ndarray:
    import pydicom
    m = np.zeros(shape, dtype=bool)
    for p in str(mask_paths).split("|"):
        a = pydicom.dcmread(p).pixel_array
        if a.shape != tuple(shape):
            a = cv2.resize(a.astype(np.uint8), (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
        m |= a > 0
    return m


BRECAHAD_KEYS = ("mitosis", "apoptosis", "tumor", "lumen")  # tubule regions = lumen


def brecahad_mask(json_path: str, shape, radius: int = 12) -> np.ndarray:
    """Union of annotated mitosis / apoptosis / tumour-nucleus / tubule points,
    each rendered as a disk of `radius` pixels (BreCaHAD ships point annotations)."""
    with open(json_path) as f:
        gt = json.load(f)
    mask = np.zeros(shape, dtype=np.uint8)
    H, W = shape
    for k in BRECAHAD_KEYS:
        for p in gt.get(k, []):
            x, y = float(p["x"]), float(p["y"])
            if x <= 1.0 and y <= 1.0:
                x, y = x * W, y * H
            cv2.circle(mask, (int(round(x)), int(round(y))), radius, 1, -1)
    return mask.astype(bool)


def load_annotation_mask(row, cfg: dict) -> Optional[np.ndarray]:
    """Mask at the native resolution of the source image."""
    if row.get("ann") in (None, "") or (isinstance(row.get("ann"), float)):
        return None
    if row["task"] == "inbreast":
        import pydicom
        ds = pydicom.dcmread(row["path"], stop_before_pixels=True)
        m = inbreast_mask(row["ann"], (int(ds.Rows), int(ds.Columns)))
    elif row["task"] == "cbis_ddsm":
        import pydicom
        ds = pydicom.dcmread(row["path"], stop_before_pixels=True)
        m = cbis_mask(row["ann"], (int(ds.Rows), int(ds.Columns)))
    elif row["task"] == "brecahad":
        h, w = read_rgb(row["path"]).shape[:2]
        m = brecahad_mask(row["ann"], (h, w), cfg.get("brecahad_point_radius", 12))
    else:
        return None
    max_side = cfg.get("mask_max_side")
    if max_side and max(m.shape) > max_side:
        s = max_side / max(m.shape)
        m = cv2.resize(m.astype(np.uint8), (int(m.shape[1] * s), int(m.shape[0] * s)),
                       interpolation=cv2.INTER_NEAREST).astype(bool)
    return m if m.any() else None
