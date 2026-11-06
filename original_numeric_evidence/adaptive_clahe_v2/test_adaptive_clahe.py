import cv2, numpy as np
import adaptive_clahe as ac

rng = np.random.default_rng(0)
H, W = 608, 968
high = rng.integers(0, 256, (H, W), dtype=np.uint8)                       # strong global + local
flat = np.clip(128 + rng.normal(0, 3, (H, W)), 0, 255).astype(np.uint8)   # low everywhere
ramp = np.tile(np.linspace(0, 255, W).astype(np.uint8), (H, 1))           # high global, low local

assert ac.decide(high)["mode"] == ac.RAW
assert ac.decide(flat)["mode"] == ac.STRONG
assert ac.decide(ramp)["mode"] == ac.MODERATE
assert ac.decide(high) == ac.decide(high)                                 # deterministic

out, d = ac.apply_adaptive_clahe(ramp)                                    # moderate == fixed baseline CLAHE
assert np.array_equal(out, cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(ramp))
assert np.array_equal(ac.apply_adaptive_clahe(high)[0], high)             # RAW untouched
print("ok", {n: ac.decide(i)["mode"] for n, i in [("high", high), ("flat", flat), ("ramp", ramp)]})

# v2: local-only gate
assert ac.decide_v2(high)["mode"] == ac.RAW
assert ac.decide_v2(flat)["mode"] == ac.MODERATE          # no strong tier in v2
assert ac.decide_v2(ramp)["mode"] == ac.MODERATE          # high global, low local -> CLAHE
assert ac.decide_v2(ramp)["mean_local_std"] < ac.LOCAL_STD_GATE_V2
out, d = ac.apply_adaptive_clahe_v2(ramp)
assert np.array_equal(out, cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(ramp))
print("v2 ok")
