"""Hand-crafted nuisance descriptors computed from tile images.

trivial16          colour / stain / focus summary (16 columns)
extended_nuisance  62 colour / focus / texture / edge features (C3)
frequency16        radial FFT band energies per RGB channel plus anisotropy (18)
morph16            Otsu-threshold shape and spatial-layout statistics (16)

The last two are cached per corpus under `nuisauc.paths.cache_dir()`.
"""
from __future__ import annotations

import gc
import logging

import numpy as np

from nuisauc.data.corpora import load_any
from nuisauc.paths import cache_dir

log = logging.getLogger(__name__)

# Ruifrok-Johnston optical-density reference vectors for the two ZN stains:
# carbol fuchsin (absorbs green strongly) and methylene blue (absorbs red).
STAIN_REF = np.array([
    [0.21, 0.87, 0.44],   # carbol fuchsin
    [0.65, 0.60, 0.47],   # methylene blue
], dtype=np.float64)
STAIN_REF /= np.linalg.norm(STAIN_REF, axis=1, keepdims=True)
STAIN_PINV = np.linalg.pinv(STAIN_REF.T)
N_BAND = 5          # radial bands, per channel, plus one anisotropy term


def stain_unmix(img01: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (fuchsin, methylene) optical-density maps."""
    od = -np.log10(np.clip(img01, 1e-3, 1.0))
    conc = od.reshape(-1, 3) @ STAIN_PINV.T
    return (conc[:, 0].reshape(img01.shape[:2]),
            conc[:, 1].reshape(img01.shape[:2]))


def trivial16(items: list) -> np.ndarray:
    """Mean and SD of R, G, B, saturation, value, fuchsin, methylene; Laplacian
    variance and grey SD."""
    from scipy.ndimage import laplace
    from skimage.color import rgb2hsv
    F = []
    for it in items:
        img = np.asarray(it["image"], dtype=np.float64) / 255.0
        hsv = rgb2hsv(img)
        fu, me = stain_unmix(img)
        gray = img.mean(2)
        ch = [img[..., 0], img[..., 1], img[..., 2],
              hsv[..., 1], hsv[..., 2], fu, me]
        F.append([f(c) for c in ch for f in (np.mean, np.std)]
                 + [float(laplace(gray).var()), gray.std()])
    return np.asarray(F)


def extended_nuisance(items: list) -> np.ndarray:
    """62 colour / focus / texture / edge features. No object morphology."""
    from scipy.ndimage import laplace
    from skimage.color import rgb2hsv
    from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
    from skimage.filters import sobel
    F = []
    for it in items:
        img = np.asarray(it["image"], dtype=np.float64) / 255.0
        hsv = rgb2hsv(img)
        fu, me = stain_unmix(img)
        gray = img.mean(2)

        ch = [img[..., 0], img[..., 1], img[..., 2],
              hsv[..., 1], hsv[..., 2], fu, me]
        row = [f(c) for c in ch for f in (np.mean, np.std)]
        row += [float(laplace(gray).var()), float(gray.std())]      # trivial-16

        for c in (img[..., 0], img[..., 1], img[..., 2], fu, me):   # +25
            row += list(np.percentile(c, [1, 10, 50, 90, 99]))

        g8 = np.clip((gray * 255), 0, 255).astype(np.uint8) // 4    # 64 levels
        glcm = graycomatrix(g8, [1], [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
                            levels=64, symmetric=True, normed=True)
        row += [float(graycoprops(glcm, p).mean()) for p in           # +6
                ("contrast", "dissimilarity", "homogeneity", "energy",
                 "correlation", "ASM")]

        lbp = local_binary_pattern((gray * 255).astype(np.uint8), 8, 1,
                                   method="uniform")
        h, _ = np.histogram(lbp, bins=10, range=(0, 10), density=True)
        row += list(h)                                                # +10

        gy, gx = np.gradient(gray)
        gm = np.hypot(gx, gy)
        row += [float(gm.mean()), float(gm.std()),                    # +3
                float(np.percentile(gm, 90))]
        row += [float(np.abs(laplace(gray)).mean()),                  # +2
                float((sobel(gray) > 0.1).mean())]
        F.append(row)
    return np.asarray(F, dtype=np.float64)


def radial_bands(gray: np.ndarray, n_band: int = N_BAND) -> list[float]:
    """Radial FFT band energy shares plus a horizontal/vertical anisotropy term."""
    f = np.fft.fftshift(np.abs(np.fft.fft2(gray)))
    h, w = f.shape
    cy, cx = h / 2.0, w / 2.0
    yy, xx = np.mgrid[0:h, 0:w]
    r = np.sqrt(((yy - cy) / max(cy, 1)) ** 2 + ((xx - cx) / max(cx, 1)) ** 2)
    tot = f.sum() + 1e-9
    edges = np.linspace(0, 1.45, n_band + 1)
    out = [float(f[(r >= a) & (r < b)].sum() / tot)
           for a, b in zip(edges[:-1], edges[1:])]
    # crude anisotropy: horizontal vs vertical energy away from DC
    mid = f.copy()
    mid[int(cy) - 1:int(cy) + 2, int(cx) - 1:int(cx) + 2] = 0.0
    hsum = float(mid[:, int(cx) - 3:int(cx) + 4].sum())
    vsum = float(mid[int(cy) - 3:int(cy) + 4, :].sum())
    out.append((hsum - vsum) / (hsum + vsum + 1e-9))
    return out


def frequency16(corpus: str, cap: int = 4000) -> np.ndarray:
    """(n, 18) radial FFT band energies per RGB channel plus anisotropy."""
    f = cache_dir() / f"freq__{corpus}__cap{cap}.npy"
    if f.exists():
        return np.load(f)
    items, _y = load_any(corpus, cap)
    rows = []
    for it in items:
        a = np.asarray(it["image"].convert("RGB"), dtype=np.float32) / 255.0
        v = []
        for ch in range(3):
            v += radial_bands(a[:, :, ch])
        rows.append(v)
    X = np.asarray(rows, dtype=np.float32)
    f.parent.mkdir(parents=True, exist_ok=True)
    np.save(f, X)
    log.info("cached %s %s", f.name, X.shape)
    del items, rows
    gc.collect()
    return X


def otsu(g: np.ndarray) -> float:
    """Otsu threshold of a [0, 1] grey image on a 64-bin histogram."""
    hist, edges = np.histogram(g, bins=64, range=(0.0, 1.0))
    hist = hist.astype(np.float64)
    w = np.cumsum(hist)
    m = np.cumsum(hist * np.arange(64))
    tot = w[-1]
    if tot == 0:
        return 0.5
    mu_t = m[-1] / tot
    w0 = w / tot
    with np.errstate(invalid="ignore", divide="ignore"):
        mu0 = m / np.maximum(w, 1e-9)
        var_b = w0 * (1 - w0) * (mu0 - mu_t) ** 2
    k = int(np.nanargmax(var_b))
    return float(edges[k + 1])


def morph_features(gray: np.ndarray) -> list[float]:
    """16 shape / layout statistics of the Otsu foreground of a grey image."""
    from scipy import ndimage
    t = otsu(gray)
    fg = gray < t                      # stained objects are darker
    lab, n = ndimage.label(fg)
    out = [float(n) / max(fg.size / 1000.0, 1e-9), float(fg.mean())]
    if n:
        areas = np.bincount(lab.ravel())[1:].astype(np.float64)
        areas /= fg.size
        out += [float(np.quantile(areas, q)) for q in (0.25, 0.5, 0.75)]
        out.append(float(areas.max()))
        out.append(float(areas.mean()))
        out.append(float(areas.std()))
    else:
        out += [0.0] * 6
    filled = ndimage.binary_fill_holes(fg)
    out.append(float(filled.mean() - fg.mean()))       # hole fraction
    h, w_ = fg.shape
    hh, hw = h // 2, w_ // 2
    quads = [fg[:hh, :hw], fg[:hh, hw:], fg[hh:, :hw], fg[hh:, hw:]]
    out += [float(q.mean()) for q in quads]
    cy0, cy1 = h // 4, 3 * h // 4
    cx0, cx1 = w_ // 4, 3 * w_ // 4
    centre = float(fg[cy0:cy1, cx0:cx1].mean())
    border = float(fg.mean())
    out.append(centre / (border + 1e-9))
    out.append(float(t))
    return out


def morph16(corpus: str, cap: int = 4000) -> np.ndarray:
    """(n, 16) morphological / spatial-layout features."""
    f = cache_dir() / f"morph__{corpus}__cap{cap}.npy"
    if f.exists():
        return np.load(f)
    items, _y = load_any(corpus, cap)
    rows = []
    for it in items:
        g = np.asarray(it["image"].convert("L"), dtype=np.float32) / 255.0
        rows.append(morph_features(g))
    X = np.asarray(rows, dtype=np.float32)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    f.parent.mkdir(parents=True, exist_ok=True)
    np.save(f, X)
    log.info("cached %s %s", f.name, X.shape)
    del items, rows
    gc.collect()
    return X
