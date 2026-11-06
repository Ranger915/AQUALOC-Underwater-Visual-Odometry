import csv
import random
from collections import defaultdict
from pathlib import Path

ROOT = Path.home() / "Downloads" / "archaeo_sequence_6_raw_data"
BASE = ROOT / "raw_data/sequence6_mutual_matching_results/sequence6_mutual_feature_metrics.csv"
ADAPT = ROOT / "sequence6_adaptive_v2_mutual_matching_results/sequence6_mutual_feature_metrics.csv"
OUT = ROOT / "sequence6_paper_validation"
OUT.mkdir(exist_ok=True)

REPS = 1000
BLOCK = 10
SEED = 20261104

def load(path, method):
    result = {}
    with path.open(newline="", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            if method == "clahe" and r["preprocessing"] != "clahe":
                continue
            key = (r["detector"], int(r["frame_id_i"]), int(r["frame_id_j"]))
            if key in result:
                raise ValueError(f"Duplicate pair: {key}")
            result[key] = {
                "label": int(r["overlap_label"]),
                "score": int(r["mutual_inliers_2px"])
            }
    return result

fixed = load(BASE, "clahe")
adaptive = load(ADAPT, "adaptive_v2")

if fixed.keys() != adaptive.keys():
    raise RuntimeError("Reference pair sets differ.")

if any(fixed[k]["label"] != adaptive[k]["label"] for k in fixed):
    raise RuntimeError("Reference labels differ.")

def f1_score(entries, weights, method):
    tp = fp = fn = 0
    for key, weight in zip(entries, weights):
        if weight == 0:
            continue
        record = method[key]
        label = record["label"]
        prediction = int(record["score"] >= 8)
        if prediction and label:
            tp += weight
        elif prediction:
            fp += weight
        elif label:
            fn += weight
    denominator = 2 * tp + fp + fn
    return 2 * tp / denominator if denominator else 0.0

def percentile(values, p):
    values = sorted(values)
    pos = (len(values) - 1) * p
    low = int(pos)
    high = min(low + 1, len(values) - 1)
    return values[low] + (values[high] - values[low]) * (pos - low)

rng = random.Random(SEED)
rows = []

for detector in ("ORB", "AKAZE_MLDB", "SIFT"):
    all_keys = sorted(k for k in fixed if k[0] == detector)
    anchors = sorted({k[1] for k in all_keys})
    anchor_index = {a: i for i, a in enumerate(anchors)}
    nanchors = len(anchors)

    for scope in ("all_pairs", "local_gap_le_100", "nonlocal_gap_ge_200"):
        keys = [
            k for k in all_keys
            if scope == "all_pairs"
            or (scope == "local_gap_le_100" and k[2] - k[1] <= 100)
            or (scope == "nonlocal_gap_ge_200" and k[2] - k[1] >= 200)
        ]
        if not keys:
            continue

        indices = [anchor_index[k[1]] for k in keys]
        ordinary = [1] * len(keys)
        point_fixed = f1_score(keys, ordinary, fixed)
        point_adaptive = f1_score(keys, ordinary, adaptive)
        differences = []

        for _ in range(REPS):
            counts = [0] * nanchors

            for _ in range((nanchors + BLOCK - 1) // BLOCK):
                start = rng.randrange(nanchors)
                for step in range(BLOCK):
                    counts[(start + step) % nanchors] += 1

            weights = [counts[i] for i in indices]

            delta = (
                f1_score(keys, weights, adaptive)
                - f1_score(keys, weights, fixed)
            )
            differences.append(delta)

        rows.append({
            "detector": detector,
            "scope": scope,
            "fixed_f1": point_fixed,
            "adaptive_f1": point_adaptive,
            "difference": point_adaptive - point_fixed,
            "ci95_lower": percentile(differences, .025),
            "ci95_upper": percentile(differences, .975),
            "probability_positive": sum(
                d > 0 for d in differences
            ) / REPS,
            "pairs": len(keys),
            "replicates": REPS,
            "block_keyframes": BLOCK,
        })

path = OUT / "adaptive_reference_f1_bootstrap.csv"
with path.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)

for r in rows:
    print(
        f'{r["detector"]:11} {r["scope"]:20} '
        f'F1 delta={r["difference"]:+.5f} '
        f'95% CI=[{r["ci95_lower"]:+.5f}, {r["ci95_upper"]:+.5f}]'
    )

print("\nSaved:", path)
