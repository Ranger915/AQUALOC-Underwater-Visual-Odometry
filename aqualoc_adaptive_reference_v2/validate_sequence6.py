import csv
from collections import Counter

p = r"C:\Users\Raaghav\Downloads\archaeo_sequence_6_raw_data\raw_data\sequence6_frontend_results\feature_pair_metrics.csv"

with open(p, "r", newline="", encoding="utf-8-sig") as f:
    reader = csv.DictReader(f)
    rows = list(reader)
    columns = reader.fieldnames or []

print("=" * 75)
print("AQUALOC SEQUENCE 06 - FULL BENCHMARK VALIDATION")
print("=" * 75)

required = [
    "detector", "preprocessing", "frame_a_id", "frame_b_id",
    "pair_stride", "keypoints_a", "keypoints_b",
    "ratio_matches", "ransac_inliers", "inlier_ratio",
    "mean_symmetric_epipolar_error_px",
    "grid_cells_covered_4x4", "status"
]

missing_columns = [c for c in required if c not in columns]
print("\nTOTAL ROWS:", len(rows))
print("MISSING REQUIRED COLUMNS:", missing_columns)

if missing_columns:
    raise SystemExit("Validation stopped: required columns are missing.")

print("\n1. CONFIGURATION COUNTS")
for key, count in sorted(Counter(
    (r["detector"], r["preprocessing"]) for r in rows
).items()):
    print(f"{key}: {count}")

print("\n2. PAIR STRIDE COUNTS")
for key, count in sorted(Counter(
    (r["detector"], r["preprocessing"], int(r["pair_stride"]))
    for r in rows
).items()):
    print(f"{key}: {count}")

print("\n3. STATUS DISTRIBUTION")
for key, count in sorted(Counter(r["status"] for r in rows).items()):
    print(f"{key}: {count}")

pair_keys = [
    (r["detector"], r["preprocessing"], int(r["pair_stride"]),
     int(r["frame_a_id"]), int(r["frame_b_id"]))
    for r in rows
]
print("\n4. DUPLICATE PAIRS:", len(pair_keys) - len(set(pair_keys)))

invalid_inliers = []
ratio_mismatches = []
stride_mismatches = []
invalid_keypoints = []
invalid_grid = []
invalid_numeric = []

for i, r in enumerate(rows, start=2):
    try:
        a = int(r["frame_a_id"])
        b = int(r["frame_b_id"])
        stride = int(r["pair_stride"])
        kp_a = int(r["keypoints_a"])
        kp_b = int(r["keypoints_b"])
        matches = int(r["ratio_matches"])
        inliers = int(r["ransac_inliers"])
        ratio = float(r["inlier_ratio"])
        grid = int(r["grid_cells_covered_4x4"])

        if inliers < 0 or inliers > matches:
            invalid_inliers.append(i)

        expected_ratio = inliers / matches if matches else 0.0
        if abs(ratio - expected_ratio) > 0.001:
            ratio_mismatches.append(i)

        if b - a != stride:
            stride_mismatches.append(i)

        if min(kp_a, kp_b, matches) < 0:
            invalid_keypoints.append(i)

        if not 0 <= grid <= 16:
            invalid_grid.append(i)

    except (ValueError, TypeError, ZeroDivisionError):
        invalid_numeric.append(i)

print("\n5. NUMERICAL VALIDATION")
print("Invalid inlier counts:", len(invalid_inliers))
print("Inlier ratio mismatches:", len(ratio_mismatches))
print("Frame stride mismatches:", len(stride_mismatches))
print("Invalid keypoint/match counts:", len(invalid_keypoints))
print("Invalid grid coverage:", len(invalid_grid))
print("Invalid numeric values:", len(invalid_numeric))

print("\n6. FRAME ID COVERAGE")
frame_ids = sorted(set(
    int(r[k]) for r in rows
    for k in ("frame_a_id", "frame_b_id")
))

if frame_ids:
    print("Minimum frame ID:", min(frame_ids))
    print("Maximum frame ID:", max(frame_ids))
    print("Unique frames referenced:", len(frame_ids))
    missing = set(range(min(frame_ids), max(frame_ids) + 1)) - set(frame_ids)
    print("Missing IDs within range:", len(missing))
else:
    print("No frame IDs found.")

print("\n7. MISSING / EMPTY VALUES")
for col in columns:
    missing = sum(
        1 for r in rows
        if r[col] is None or r[col].strip() == ""
    )
    if missing:
        print(f"{col}: {missing}")

print("\n8. FAILED PAIR BREAKDOWN")
failed = [r for r in rows if r["status"].strip().lower() != "ok"]
print("Total non-OK pairs:", len(failed))

for key, count in sorted(Counter(
    (r["detector"], r["preprocessing"], r["pair_stride"], r["status"])
    for r in failed
).items()):
    print(f"{key}: {count}")

print("\n9. EXPECTED VS ACTUAL ROW COUNT")
expected_per_config = 3384 + 3380 + 3365
expected_total = expected_per_config * 6

print("Expected stride 1 pairs: 3384")
print("Expected stride 5 pairs: 3380")
print("Expected stride 20 pairs: 3365")
print("Expected rows per configuration:", expected_per_config)
print("Expected total rows:", expected_total)
print("Actual total rows:", len(rows))
print("ROW COUNT MATCH:", len(rows) == expected_total)

print("\n" + "=" * 75)
print("VALIDATION COMPLETE")
print("=" * 75)
