import os
import math
import numpy as np

root = os.path.join(
    os.path.expanduser("~"),
    "Downloads",
    "archaeo_sequence_6_raw_data"
)

paths = [
    os.path.join(root, "colmap_traj_sequence_6.txt"),
    os.path.join(root, "new_archaeo_colmap_traj_sequence_06.txt")
]

def load(path):
    result = {}
    with open(path, "r", encoding="utf-8-sig") as f:
        for line in f:
            if not line.strip():
                continue
            v = [float(x) for x in line.split()]
            if len(v) == 8:
                result[int(v[0])] = v[1:]
    return result

a = load(paths[0])
b = load(paths[1])

common = sorted(set(a) & set(b))

# Compare the three translation columns at common frame IDs.
# Fit: trajectory_2 * scale * rotation + translation ~= trajectory_1
X = np.array([b[i][:3] for i in common], dtype=float)
Y = np.array([a[i][:3] for i in common], dtype=float)

mx = X.mean(axis=0)
my = Y.mean(axis=0)

Xc = X - mx
Yc = Y - my

H = Xc.T @ Yc / len(common)
U, singular_values, Vt = np.linalg.svd(H)

D = np.eye(3)
D[2, 2] = np.linalg.det(U @ Vt)

R = U @ D @ Vt
variance_x = np.sum(Xc ** 2) / len(common)
scale = np.sum(singular_values * np.diag(D)) / variance_x

translation = my - scale * (mx @ R)

X_aligned = scale * (X @ R) + translation
residuals = np.linalg.norm(X_aligned - Y, axis=1)

print("=" * 72)
print("AQUALOC SEQUENCE 06 - SIMILARITY ALIGNMENT")
print("=" * 72)
print("Common frame IDs:", len(common))
print("Source trajectory: trajectory 2")
print("Target trajectory: trajectory 1")
print()
print("Estimated scale:", scale)
print("Estimated rotation matrix:")
for row in R:
    print(" ", [float(x) for x in row])
print("Estimated translation:", translation.tolist())
print()
print("POSITION RESIDUALS AFTER SIM(3) ALIGNMENT")
print("RMSE:", float(np.sqrt(np.mean(residuals ** 2))))
print("Mean:", float(np.mean(residuals)))
print("Median:", float(np.median(residuals)))
print("95th percentile:", float(np.percentile(residuals, 95)))
print("Maximum:", float(np.max(residuals)))
print()
print("FIRST 15 FRAME RESIDUALS")
for fid, err in list(zip(common, residuals))[:15]:
    print(f"Frame {fid:4d}: {err:.9f}")
print()
print("ALIGNMENT DIAGNOSTICS")
print("Target trajectory spatial spread:",
      float(np.sqrt(np.mean(np.sum(Yc ** 2, axis=1)))))
print("Residual / target spread:",
      float(np.sqrt(np.mean(residuals ** 2)) /
            max(np.sqrt(np.mean(np.sum(Yc ** 2, axis=1))), 1e-12)))
print("=" * 72)
