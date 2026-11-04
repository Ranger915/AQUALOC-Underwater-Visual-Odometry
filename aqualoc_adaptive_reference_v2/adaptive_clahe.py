"""Deterministic, image-only Adaptive CLAHE for AQUALOC Sequence 06.

V1: high contrast if global_std>=35 AND mean_local_std>=15.
V2: inferred from research notes: high contrast if mean_local_std>=10.5.
Both: strong only when global_std<25 AND mean_local_std<10.
IMPORTANT: confirm the exact v2 decision expression against your existing v2 code.
"""
from dataclasses import dataclass, asdict
import cv2
import numpy as np

@dataclass(frozen=True)
class AdaptiveConfig:
    version: str = "v2"
    global_threshold: float = 35.0
    local_threshold_v1: float = 15.0
    local_threshold_v2: float = 10.5
    severe_global_threshold: float = 25.0
    severe_local_threshold: float = 10.0
    moderate_clip: float = 2.0
    strong_clip: float = 3.0
    grid: int = 8


def measure_contrast(gray, grid=8):
    if gray.ndim != 2 or gray.dtype != np.uint8:
        raise ValueError("Expected a grayscale uint8 image")
    h, w = gray.shape
    if h < grid or w < grid:
        raise ValueError("Image smaller than contrast measurement grid")
    global_std = float(np.std(gray.astype(np.float64)))
    ys = np.linspace(0, h, grid + 1, dtype=int)
    xs = np.linspace(0, w, grid + 1, dtype=int)
    local_std = float(np.mean([
        np.std(gray[ys[i]:ys[i+1], xs[j]:xs[j+1]].astype(np.float64))
        for i in range(grid) for j in range(grid)
    ]))
    return global_std, local_std


def decide(gray, config=AdaptiveConfig()):
    g, l = measure_contrast(gray, config.grid)
    if config.version == "v1":
        keep_raw = g >= config.global_threshold and l >= config.local_threshold_v1
    elif config.version == "v2":
        keep_raw = l >= config.local_threshold_v2
    else:
        raise ValueError("version must be v1 or v2")
    if keep_raw:
        mode, clip = "raw", 0.0
    elif g < config.severe_global_threshold and l < config.severe_local_threshold:
        mode, clip = "strong_clahe", config.strong_clip
    else:
        mode, clip = "moderate_clahe", config.moderate_clip
    return {"global_std": g, "local_std": l, "decision": mode,
            "clip_limit": clip, "version": config.version}


def apply_adaptive(gray, config=AdaptiveConfig(), decision=None):
    decision = decide(gray, config) if decision is None else decision
    if decision["decision"] == "raw":
        return gray, decision
    enhanced = cv2.createCLAHE(clipLimit=decision["clip_limit"],
                              tileGridSize=(config.grid, config.grid)).apply(gray)
    return enhanced, decision
