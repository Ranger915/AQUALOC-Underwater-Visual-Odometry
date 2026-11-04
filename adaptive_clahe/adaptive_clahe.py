"""Adaptive CLAHE preprocessing for the AQUALOC feature benchmark.

The decision uses only the current image's intensity statistics: no COLMAP data,
ground truth, matches or RANSAC results. Import this module from other scripts so
every experiment uses the exact same implementation.

Fixed experimental configuration (v1). Do not tune against evaluation results.
"""
import cv2
import numpy as np

GLOBAL_STD_NORMAL = 35.0   # global std >= this (and local ok) -> RAW
GLOBAL_STD_SEVERE = 25.0   # global std < this (and local severe) -> strong
LOCAL_STD_NORMAL = 15.0
LOCAL_STD_SEVERE = 10.0
STATS_GRID = 8             # local std measured on an 8 x 8 grid
CLIP_MODERATE = 2.0
CLIP_STRONG = 3.0
TILE_GRID = (8, 8)         # CLAHE tile grid, same as the fixed-CLAHE baseline

RAW, MODERATE, STRONG = "raw", "moderate_clahe", "strong_clahe"


def config():
    return {
        "global_std_normal": GLOBAL_STD_NORMAL,
        "global_std_severe": GLOBAL_STD_SEVERE,
        "local_std_normal": LOCAL_STD_NORMAL,
        "local_std_severe": LOCAL_STD_SEVERE,
        "stats_grid": STATS_GRID,
        "clip_moderate": CLIP_MODERATE,
        "clip_strong": CLIP_STRONG,
        "clahe_tile_grid": list(TILE_GRID),
    }


def contrast_stats(gray):
    """Return (global std, mean of per-cell std over a STATS_GRID x STATS_GRID grid)."""
    h, w = gray.shape
    ys = np.linspace(0, h, STATS_GRID + 1).astype(int)
    xs = np.linspace(0, w, STATS_GRID + 1).astype(int)
    local = [
        gray[y0:y1, x0:x1].std()
        for y0, y1 in zip(ys[:-1], ys[1:])
        for x0, x1 in zip(xs[:-1], xs[1:])
        if y1 > y0 and x1 > x0
    ]
    return float(gray.std()), float(np.mean(local))


def decide(gray):
    """Per-image decision record. Depends only on gray's own statistics."""
    g, l = contrast_stats(gray)
    if g >= GLOBAL_STD_NORMAL and l >= LOCAL_STD_NORMAL:
        mode, clip = RAW, None
    elif g < GLOBAL_STD_SEVERE and l < LOCAL_STD_SEVERE:
        mode, clip = STRONG, CLIP_STRONG
    else:
        # Spec rule 4 ("otherwise RAW") is unreachable: once the RAW test fails,
        # at least one statistic is below its normal threshold, so this is moderate.
        mode, clip = MODERATE, CLIP_MODERATE
    return {
        "global_std": round(g, 6),
        "mean_local_std": round(l, 6),
        "mode": mode,
        "clip_limit": clip,
        "tile_grid": None if clip is None else f"{TILE_GRID[0]}x{TILE_GRID[1]}",
    }


def enhance(gray, clip_limit):
    """Apply CLAHE with the given clip limit, or return gray unchanged for RAW (None)."""
    if clip_limit is None:
        return gray
    return cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=TILE_GRID).apply(gray)


def apply_adaptive_clahe(gray):
    """Convenience entry point for other scripts: returns (image, decision record)."""
    d = decide(gray)
    return enhance(gray, d["clip_limit"]), d
