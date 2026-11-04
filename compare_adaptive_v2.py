import csv
import math
from collections import defaultdict
from pathlib import Path

ROOT = Path(r"C:\Users\Raaghav\Downloads\archaeo_sequence_6_raw_data")

BASELINE = ROOT / "raw_data" / "sequence6_frontend_results" / "feature_pair_metrics.csv"
ADAPTIVE = ROOT / "sequence6_adaptive_v2_full" / "adaptive_feature_pair_metrics.csv"
OUT = ROOT / "sequence6_adaptive_v2_comparison"
OUT.mkdir(parents=True, exist_ok=True)

def number(value):
    try:
        result = float(value)
        return result if math.isfinite(result) else None
    except (TypeError, ValueError):
        return None

def normalized_detector(value):
    value = value.strip().upper()
    return "AKAZE_MLDB" if value == "AKAZE" else value

def load(path, adaptive=False):
    data = {}
    with path.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)

        # Normalize column names from the original benchmark.
        aliases = {
            "frame_a_id": "frame_id_1",
            "frame_b_id": "frame_id_2",
            "pair_stride": "frame_stride",
        }

        reader.fieldnames = [
            aliases.get(name.strip(), name.strip())
            for name in reader.fieldnames
        ]
        required = {
            "detector", "frame_id_1", "frame_id_2",
            "frame_stride", "status", "ransac_inliers",
            "ratio_matches", "inlier_ratio"
        }
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path}: missing columns {sorted(missing)}")

        for row in reader:
            method = (
                "adaptive_v2"
                if adaptive else row["preprocessing"].strip().lower()
            )
            detector = normalized_detector(row["detector"])
            key = (
                detector,
                int(row["frame_id_1"]),
                int(row["frame_id_2"]),
                int(row["frame_stride"])
            )
            unique_key = (method, *key)
            if unique_key in data:
                raise ValueError(f"Duplicate record: {unique_key}")
            data[unique_key] = row
    return data

baseline = load(BASELINE)
adaptive = load(ADAPTIVE, adaptive=True)
all_rows = {**baseline, **adaptive}

print("Baseline rows:", len(baseline))
print("Adaptive rows:", len(adaptive))

groups = defaultdict(dict)
for key, row in all_rows.items():
    method, detector, a, b, stride = key
    groups[(detector, a, b, stride)][method] = row

methods = ("raw", "clahe", "adaptive_v2")
complete = {
    key: val for key, val in groups.items()
    if all(m in val for m in methods)
}

print("Complete matched detector/pair groups:", len(complete))

expected = 30387
if len(complete) != expected:
    print(
        "WARNING: Expected 30387 complete groups. "
        "Inspect missing pairs before interpreting results."
    )

def mean_valid(values):
    values = [x for x in values if x is not None]
    return sum(values) / len(values) if values else None

def summarize(rows):
    n = len(rows)
    successful = [r for r in rows if r["status"] == "ok"]
    errors = [
        number(r.get("mean_symmetric_epipolar_error_px"))
        for r in successful
    ]
    return {
        "pairs": n,
        "successful_pairs": len(successful),
        "success_rate_pct": 100 * len(successful) / n if n else None,
        "mean_ratio_matches": mean_valid([
            number(r["ratio_matches"]) for r in rows
        ]),
        "mean_ransac_inliers": mean_valid([
            number(r["ransac_inliers"]) for r in rows
        ]),
        "mean_inlier_ratio_all": mean_valid([
            number(r["inlier_ratio"]) for r in rows
        ]),
        "mean_inlier_ratio_success_only": mean_valid([
            number(r["inlier_ratio"]) for r in successful
        ]),
        "mean_epipolar_error_ok_px": mean_valid(errors),
        "mean_matching_time_s": mean_valid([
            number(r.get("matching_time_s")) for r in rows
        ]),
        "mean_geometry_time_s": mean_valid([
            number(r.get("geometry_time_s")) for r in rows
        ])
    }

summary = []
for detector in ("ORB", "AKAZE_MLDB", "SIFT"):
    for stride in ("all", 1, 5, 20):
        selected = [
            (key, vals) for key, vals in complete.items()
            if key[0] == detector
            and (stride == "all" or key[3] == stride)
        ]
        for method in methods:
            result = summarize([vals[method] for _, vals in selected])
            summary.append({
                "detector": detector,
                "stride": stride,
                "method": method,
                **result
            })

summary_path = OUT / "baseline_vs_adaptive_v2_summary.csv"

with summary_path.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=summary[0].keys())
    writer.writeheader()
    writer.writerows(summary)

def difference(a, b, column):
    x = number(a.get(column))
    y = number(b.get(column))
    if x is None or y is None:
        return None
    return x - y

comparisons = []
for detector in ("ORB", "AKAZE_MLDB", "SIFT"):
    for stride in ("all", 1, 5, 20):
        selected = [
            vals for key, vals in complete.items()
            if key[0] == detector
            and (stride == "all" or key[3] == stride)
        ]

        for reference in ("raw", "clahe"):
            success_changes = [
                int(v["adaptive_v2"]["status"] == "ok")
                - int(v[reference]["status"] == "ok")
                for v in selected
            ]
            inlier_changes = [
                difference(
                    v["adaptive_v2"],
                    v[reference],
                    "ransac_inliers"
                )
                for v in selected
            ]
            ratio_changes = [
                difference(
                    v["adaptive_v2"],
                    v[reference],
                    "ratio_matches"
                )
                for v in selected
            ]
            epipolar_changes = [
                difference(
                    v["adaptive_v2"],
                    v[reference],
                    "mean_symmetric_epipolar_error_px"
                )
                for v in selected
                if v["adaptive_v2"]["status"] == "ok"
                and v[reference]["status"] == "ok"
            ]
            comparisons.append({
                "detector": detector,
                "stride": stride,
                "comparison": f"adaptive_v2_minus_{reference}",
                "matched_pairs": len(selected),
                "success_rate_change_pp": (
                    100 * sum(success_changes) / len(selected)
                    if selected else None
                ),
                "mean_ratio_match_difference": mean_valid(ratio_changes),
                "mean_ransac_inlier_difference": mean_valid(inlier_changes),
                "mean_epipolar_error_difference_px": mean_valid(
                    epipolar_changes
                ),
                "epipolar_common_success_pairs": sum(
                    x is not None for x in epipolar_changes
                )
            })

comparison_path = OUT / "adaptive_v2_paired_differences.csv"

with comparison_path.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=comparisons[0].keys())
    writer.writeheader()
    writer.writerows(comparisons)

print("\n" + "=" * 78)
print("COMPARISON: ALL THREE FRAME STRIDES")
print("=" * 78)

for row in summary:
    if row["stride"] != "all":
        continue
    print(
        f'{row["detector"]:12} '
        f'{row["method"]:12} '
        f'Success={row["success_rate_pct"]:6.2f}% '
        f'Inliers={row["mean_ransac_inliers"]:8.2f} '
        f'Ratio={row["mean_inlier_ratio_all"]:.4f} '
        f'Epipolar={row["mean_epipolar_error_ok_px"]}'
    )

print("\n" + "=" * 78)
print("PAIRED DIFFERENCES: ADAPTIVE v2 MINUS BASELINE")
print("=" * 78)

for row in comparisons:
    if row["stride"] != "all":
        continue
    print(
        f'{row["detector"]:12} '
        f'{row["comparison"]:27} '
        f'SuccessDelta={row["success_rate_change_pp"]:+7.3f} pp '
        f'InlierDelta={row["mean_ransac_inlier_difference"]:+9.2f}'
    )

print("\nFiles saved:")
print(summary_path)
print(comparison_path)
