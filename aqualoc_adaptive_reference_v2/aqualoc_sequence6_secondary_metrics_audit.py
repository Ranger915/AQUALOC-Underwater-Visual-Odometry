import csv
import json
import math
import statistics
from pathlib import Path
from collections import Counter

ROOT = Path.home() / "Downloads" / "archaeo_sequence_6_raw_data"

RESULT_DIR = (
    ROOT / "raw_data" / "sequence6_calibrated_feature_results"
)

METRICS = RESULT_DIR / "sequence6_reference_feature_metrics.csv"
METADATA = RESULT_DIR / "sequence6_reference_feature_metadata.json"

OUT = RESULT_DIR

CONFIGS = [
    ("ORB", "raw"),
    ("ORB", "clahe"),
    ("AKAZE_MLDB", "raw"),
    ("AKAZE_MLDB", "clahe"),
    ("SIFT", "raw"),
    ("SIFT", "clahe")
]

SCOPES = [
    ("all_pairs", lambda r: True),
    ("local_gap_le_100", lambda r: r["frame_gap"] <= 100),
    ("nonlocal_gap_ge_200", lambda r: r["frame_gap"] >= 200)
]

BASELINE_BINS = [
    ("<0.05", 0.0, 0.05),
    ("0.05-0.10", 0.05, 0.10),
    ("0.10-0.25", 0.10, 0.25),
    ("0.25-0.50", 0.25, 0.50),
    ("0.50-1.00", 0.50, 1.00),
    ">=1.00"
]


def number(row, key):
    value = row.get(key, "")
    if value in ("", None):
        return 0.0
    return float(value)


def average_precision(labels, scores):
    labels = [int(x) for x in labels]
    scores = [float(x) for x in scores]

    positives = sum(labels)

    if positives == 0:
        return float("nan")

    ordered = sorted(
        zip(scores, labels),
        key=lambda x: -x[0]
    )

    tp = 0
    fp = 0
    previous_recall = 0.0
    ap = 0.0

    index = 0

    while index < len(ordered):
        score = ordered[index][0]
        group_labels = []

        while (
            index < len(ordered)
            and ordered[index][0] == score
        ):
            group_labels.append(ordered[index][1])
            index += 1

        tp += sum(group_labels)
        fp += len(group_labels) - sum(group_labels)

        recall = tp / positives
        precision = tp / (tp + fp)

        ap += (recall - previous_recall) * precision
        previous_recall = recall

    return ap


def classification(labels, predictions):
    tp = sum(
        1 for y, p in zip(labels, predictions)
        if y == 1 and p == 1
    )

    fp = sum(
        1 for y, p in zip(labels, predictions)
        if y == 0 and p == 1
    )

    fn = sum(
        1 for y, p in zip(labels, predictions)
        if y == 1 and p == 0
    )

    tn = sum(
        1 for y, p in zip(labels, predictions)
        if y == 0 and p == 0
    )

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0

    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall else 0.0
    )

    return tp, fp, fn, tn, precision, recall, f1


def mean_or_zero(values):
    return statistics.mean(values) if values else 0.0


def median_or_zero(values):
    return statistics.median(values) if values else 0.0


def percentile(values, p):
    if not values:
        return 0.0

    ordered = sorted(values)
    position = (len(ordered) - 1) * p / 100.0
    lower = int(math.floor(position))
    upper = int(math.ceil(position))

    if lower == upper:
        return ordered[lower]

    fraction = position - lower

    return (
        ordered[lower] * (1.0 - fraction)
        + ordered[upper] * fraction
    )


def baseline_bin(value):
    if value < 0.05:
        return "<0.05"
    if value < 0.10:
        return "0.05-0.10"
    if value < 0.25:
        return "0.10-0.25"
    if value < 0.50:
        return "0.25-0.50"
    if value < 1.00:
        return "0.50-1.00"
    return ">=1.00"


def write_csv(path, rows):
    if not rows:
        print("[WARN] No rows to write:", path)
        return

    fields = list(rows[0].keys())

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    if not METRICS.is_file():
        raise FileNotFoundError(METRICS)

    with METADATA.open("r", encoding="utf-8") as f:
        metadata = json.load(f)

    rows = []

    with METRICS.open("r", newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            rows.append({
                **row,
                "frame_gap": int(row["frame_gap"]),
                "overlap_label": int(row["overlap_label"]),
                "baseline": number(row, "baseline_export_units"),
                "inliers_2px": number(row, "reference_inliers_2px"),
                "fraction_2px": number(
                    row, "reference_inlier_ratio_2px"
                ),
                "ratio_matches": number(row, "ratio_matches"),
                "matching_time": number(row, "matching_time_s"),
                "geometry_time": number(row, "geometry_time_s")
            })

    expected = 14365 * 6

    print("=" * 84)
    print("AQUALOC SEQUENCE 06 - SECONDARY METRICS AND RELIABILITY AUDIT")
    print("=" * 84)

    print("\n1. ROW INTEGRITY")
    print("-" * 84)
    print("Rows found:", len(rows))
    print("Expected:", expected)
    print("Integrity:", "PASS" if len(rows) == expected else "CHECK")

    # --------------------------------------------------------------
    # A. Compare absolute and normalized retrieval-ranking scores.
    # --------------------------------------------------------------

    ranking_rows = []

    print("\n2. ABSOLUTE VS NORMALIZED AVERAGE PRECISION")
    print("-" * 84)
    print(
        f"{'Detector':<13}{'Prep':<8}{'Scope':<23}"
        f"{'AP count':>11}{'AP fraction':>14}"
    )

    for detector, preprocessing in CONFIGS:
        config = [
            r for r in rows
            if r["detector"] == detector
            and r["preprocessing"] == preprocessing
        ]

        for scope_name, condition in SCOPES:
            selected = [r for r in config if condition(r)]

            labels = [r["overlap_label"] for r in selected]

            ap_count = average_precision(
                labels,
                [r["inliers_2px"] for r in selected]
            )

            ap_fraction = average_precision(
                labels,
                [r["fraction_2px"] for r in selected]
            )

            print(
                f"{detector:<13}{preprocessing:<8}{scope_name:<23}"
                f"{ap_count:>11.4f}{ap_fraction:>14.4f}"
            )

            ranking_rows.append({
                "detector": detector,
                "preprocessing": preprocessing,
                "scope": scope_name,
                "average_precision_absolute_count_2px": ap_count,
                "average_precision_normalized_fraction_2px": ap_fraction,
                "pair_count": len(selected),
                "positive_pairs": sum(labels)
            })

    # --------------------------------------------------------------
    # B. Fixed operating-point metrics.
    # --------------------------------------------------------------

    operating_rows = []

    print("\n3. FIXED OPERATING POINTS: 2-PIXEL SAMPSON THRESHOLD")
    print("-" * 84)
    print(
        f"{'Detector':<13}{'Prep':<8}{'Scope':<23}"
        f"{'Min':>5}{'Prec':>10}{'Recall':>10}{'F1':>10}"
    )

    for detector, preprocessing in CONFIGS:
        config = [
            r for r in rows
            if r["detector"] == detector
            and r["preprocessing"] == preprocessing
        ]

        for scope_name, condition in SCOPES:
            selected = [r for r in config if condition(r)]

            labels = [r["overlap_label"] for r in selected]

            for minimum in (8, 12, 20):
                predictions = [
                    int(r["inliers_2px"] >= minimum)
                    for r in selected
                ]

                (
                    tp, fp, fn, tn,
                    precision, recall, f1
                ) = classification(labels, predictions)

                print(
                    f"{detector:<13}{preprocessing:<8}{scope_name:<23}"
                    f"{minimum:>5}{precision:>10.4f}"
                    f"{recall:>10.4f}{f1:>10.4f}"
                )

                operating_rows.append({
                    "detector": detector,
                    "preprocessing": preprocessing,
                    "scope": scope_name,
                    "sampson_threshold_px": 2.0,
                    "minimum_reference_inliers": minimum,
                    "pair_count": len(selected),
                    "positive_pairs": sum(labels),
                    "negative_pairs": len(labels) - sum(labels),
                    "TP": tp,
                    "FP": fp,
                    "FN": fn,
                    "TN": tn,
                    "precision": precision,
                    "recall": recall,
                    "f1": f1
                })

    # --------------------------------------------------------------
    # C. Baseline-stratified positive-pair recall.
    # --------------------------------------------------------------

    baseline_rows = []

    print("\n4. POSITIVE-PAIR RECALL BY BASELINE")
    print("-" * 84)
    print(
        f"{'Detector':<13}{'Prep':<8}{'Baseline':<13}"
        f"{'Positive':>10}{'Detected':>10}{'Recall@8':>12}"
    )

    for detector, preprocessing in CONFIGS:
        config = [
            r for r in rows
            if r["detector"] == detector
            and r["preprocessing"] == preprocessing
            and r["overlap_label"] == 1
        ]

        for bin_spec in BASELINE_BINS:
            name = bin_spec[0] if isinstance(bin_spec, tuple) else bin_spec

            selected = [
                r for r in config
                if baseline_bin(r["baseline"]) == name
            ]

            detected = sum(
                1 for r in selected if r["inliers_2px"] >= 8
            )

            recall = detected / len(selected) if selected else 0.0

            print(
                f"{detector:<13}{preprocessing:<8}{name:<13}"
                f"{len(selected):>10}{detected:>10}{recall:>11.4f}"
            )

            baseline_rows.append({
                "detector": detector,
                "preprocessing": preprocessing,
                "baseline_bin": name,
                "positive_pairs": len(selected),
                "detected_at_8_inliers_2px": detected,
                "positive_pair_recall": recall,
                "mean_reference_inliers_2px": mean_or_zero(
                    [r["inliers_2px"] for r in selected]
                ),
                "median_reference_inliers_2px": median_or_zero(
                    [r["inliers_2px"] for r in selected]
                )
            })

    # --------------------------------------------------------------
    # D. Matching and geometry scoring time.
    # --------------------------------------------------------------

    timing_rows = []

    print("\n5. MATCHING AND REFERENCE-GEOMETRY SCORING TIME")
    print("-" * 84)
    print(
        f"{'Detector':<13}{'Prep':<8}{'Match total(s)':>16}"
        f"{'Match P95(ms)':>16}{'Score total(s)':>16}"
        f"{'Score P95(ms)':>16}"
    )

    for detector, preprocessing in CONFIGS:
        config = [
            r for r in rows
            if r["detector"] == detector
            and r["preprocessing"] == preprocessing
        ]

        match_times = [r["matching_time"] for r in config]
        geometry_times = [r["geometry_time"] for r in config]

        total_match = sum(match_times)
        total_geometry = sum(geometry_times)

        match_p95 = percentile(match_times, 95) * 1000
        geometry_p95 = percentile(geometry_times, 95) * 1000

        print(
            f"{detector:<13}{preprocessing:<8}"
            f"{total_match:>16.4f}{match_p95:>16.4f}"
            f"{total_geometry:>16.4f}{geometry_p95:>16.4f}"
        )

        timing_rows.append({
            "detector": detector,
            "preprocessing": preprocessing,
            "pair_count": len(config),
            "matching_time_total_s": total_match,
            "matching_time_mean_ms": mean_or_zero(match_times) * 1000,
            "matching_time_p95_ms": match_p95,
            "geometry_scoring_time_total_s": total_geometry,
            "geometry_scoring_time_mean_ms": (
                mean_or_zero(geometry_times) * 1000
            ),
            "geometry_scoring_time_p95_ms": geometry_p95
        })

    # --------------------------------------------------------------
    # E. Status and feature-support diagnostics.
    # --------------------------------------------------------------

    status_rows = []

    print("\n6. ZERO-MATCH / STATUS AUDIT")
    print("-" * 84)

    for detector, preprocessing in CONFIGS:
        config = [
            r for r in rows
            if r["detector"] == detector
            and r["preprocessing"] == preprocessing
        ]

        statuses = Counter(r.get("status", "") for r in config)
        zero_matches = sum(
            1 for r in config if r["ratio_matches"] == 0
        )

        positives = [r for r in config if r["overlap_label"] == 1]
        negatives = [r for r in config if r["overlap_label"] == 0]

        print(
            f"{detector:<13}{preprocessing:<8}"
            f"zero ratio-match pairs={zero_matches:>5}, "
            f"statuses={dict(statuses)}"
        )

        status_rows.append({
            "detector": detector,
            "preprocessing": preprocessing,
            "zero_ratio_match_pairs": zero_matches,
            "positive_pair_count": len(positives),
            "negative_pair_count": len(negatives),
            "positive_mean_ratio_matches": mean_or_zero(
                [r["ratio_matches"] for r in positives]
            ),
            "negative_mean_ratio_matches": mean_or_zero(
                [r["ratio_matches"] for r in negatives]
            ),
            "positive_mean_reference_inliers_2px": mean_or_zero(
                [r["inliers_2px"] for r in positives]
            ),
            "negative_mean_reference_inliers_2px": mean_or_zero(
                [r["inliers_2px"] for r in negatives]
            ),
            "status_counts": json.dumps(dict(statuses))
        })

    # --------------------------------------------------------------
    # F. Save outputs.
    # --------------------------------------------------------------

    paths = {
        "ranking": OUT / "sequence6_secondary_ranking_audit.csv",
        "operating": OUT / "sequence6_secondary_operating_points.csv",
        "baseline": OUT / "sequence6_secondary_baseline_audit.csv",
        "timing": OUT / "sequence6_secondary_timing_audit.csv",
        "status": OUT / "sequence6_secondary_status_audit.csv"
    }

    write_csv(paths["ranking"], ranking_rows)
    write_csv(paths["operating"], operating_rows)
    write_csv(paths["baseline"], baseline_rows)
    write_csv(paths["timing"], timing_rows)
    write_csv(paths["status"], status_rows)

    print("\n7. OUTPUT FILES")
    print("-" * 84)

    for name, path in paths.items():
        print(name, ":", path)

    print("\nOriginal benchmark files modified: NO")
    print("Calibrated reference files modified: NO")
    print("=" * 84)


if __name__ == "__main__":
    main()
