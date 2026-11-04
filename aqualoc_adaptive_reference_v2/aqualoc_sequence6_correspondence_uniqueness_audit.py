import csv
import gzip
from pathlib import Path
from collections import defaultdict

ROOT = Path.home() / "Downloads" / "archaeo_sequence_6_raw_data"

RESULT_DIR = ROOT / "raw_data" / "sequence6_calibrated_feature_results"

METRICS_FILE = RESULT_DIR / "sequence6_reference_feature_metrics.csv"

CONFIGS = [
    ("ORB", "raw"),
    ("ORB", "clahe"),
    ("AKAZE_MLDB", "raw"),
    ("AKAZE_MLDB", "clahe"),
    ("SIFT", "raw"),
    ("SIFT", "clahe")
]

SCOPES = [
    ("all_pairs", lambda gap: True),
    ("local_gap_le_100", lambda gap: gap <= 100),
    ("nonlocal_gap_ge_200", lambda gap: gap >= 200)
]


def classification(labels, predictions):
    tp = sum(y == 1 and p == 1 for y, p in zip(labels, predictions))
    fp = sum(y == 0 and p == 1 for y, p in zip(labels, predictions))
    fn = sum(y == 1 and p == 0 for y, p in zip(labels, predictions))
    tn = sum(y == 0 and p == 0 for y, p in zip(labels, predictions))

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    return tp, fp, fn, tn, precision, recall, f1


def load_metrics():
    data = {}

    with METRICS_FILE.open("r", newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            key = (
                row["detector"],
                row["preprocessing"],
                int(row["frame_id_i"]),
                int(row["frame_id_j"])
            )

            data[key] = {
                "gap": int(row["frame_gap"]),
                "label": int(row["overlap_label"]),
                "ratio_matches": int(row["ratio_matches"]),
                "reference_inliers_2px": int(row["reference_inliers_2px"])
            }

    return data


def audit_configuration(detector, preprocessing, metrics):
    filename = f"matches_{detector}_{preprocessing}.csv.gz"
    path = RESULT_DIR / filename

    if not path.is_file():
        raise FileNotFoundError(path)

    # Per-pair correspondence statistics
    stats = defaultdict(lambda: {
        "total": 0,
        "inliers": 0,
        "query_ids": set(),
        "train_ids": set(),
        "inlier_train_ids": set()
    })

    with gzip.open(path, "rt", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for row in reader:
            key = (
                detector,
                preprocessing,
                int(row["frame_id_i"]),
                int(row["frame_id_j"])
            )

            item = stats[key]

            query_id = int(row["query_keypoint_index"])
            train_id = int(row["train_keypoint_index"])
            error = float(row["sampson_distance_px"])

            item["total"] += 1
            item["query_ids"].add(query_id)
            item["train_ids"].add(train_id)

            if error <= 2.0:
                item["inliers"] += 1
                item["inlier_train_ids"].add(train_id)

    # Validate correspondence file against metric file.
    count_mismatches = 0
    inlier_mismatches = 0

    for key, m in metrics.items():
        if key[0] != detector or key[1] != preprocessing:
            continue

        item = stats.get(key)

        total = item["total"] if item else 0
        inliers = item["inliers"] if item else 0

        if total != m["ratio_matches"]:
            count_mismatches += 1

        if inliers != m["reference_inliers_2px"]:
            inlier_mismatches += 1

    if count_mismatches or inlier_mismatches:
        raise ValueError(
            f"{detector}/{preprocessing}: correspondence mismatch "
            f"(matches={count_mismatches}, inliers={inlier_mismatches})"
        )

    output_rows = []

    for scope_name, condition in SCOPES:
        selected = []

        for key, m in metrics.items():
            if key[0] != detector or key[1] != preprocessing:
                continue

            if not condition(m["gap"]):
                continue

            item = stats.get(key)

            total = item["total"] if item else 0
            inliers = item["inliers"] if item else 0

            unique_query = len(item["query_ids"]) if item else 0
            unique_train = len(item["train_ids"]) if item else 0
            unique_inlier_train = len(item["inlier_train_ids"]) if item else 0

            selected.append({
                "label": m["label"],
                "raw_matches": total,
                "raw_inliers": inliers,
                "unique_query": unique_query,
                "unique_train": unique_train,
                "unique_inlier_train": unique_inlier_train
            })

        labels = [x["label"] for x in selected]

        raw_predictions = [
            int(x["raw_inliers"] >= 8) for x in selected
        ]

        unique_predictions = [
            int(x["unique_inlier_train"] >= 8) for x in selected
        ]

        raw_result = classification(labels, raw_predictions)
        unique_result = classification(labels, unique_predictions)

        total_matches = sum(x["raw_matches"] for x in selected)
        unique_matches = sum(x["unique_train"] for x in selected)

        total_inliers = sum(x["raw_inliers"] for x in selected)
        unique_inliers = sum(x["unique_inlier_train"] for x in selected)

        output_rows.append({
            "detector": detector,
            "preprocessing": preprocessing,
            "scope": scope_name,
            "pair_count": len(selected),
            "ratio_matches_total": total_matches,
            "unique_train_keypoints_total": unique_matches,
            "train_collision_percentage": (
                100 * (total_matches - unique_matches) / total_matches
                if total_matches else 0.0
            ),
            "reference_inliers_2px_total": total_inliers,
            "unique_train_reference_inliers_total": unique_inliers,
            "inlier_collision_percentage": (
                100 * (total_inliers - unique_inliers) / total_inliers
                if total_inliers else 0.0
            ),
            "raw_TP": raw_result[0],
            "raw_FP": raw_result[1],
            "raw_FN": raw_result[2],
            "raw_TN": raw_result[3],
            "raw_precision": raw_result[4],
            "raw_recall": raw_result[5],
            "raw_f1": raw_result[6],
            "unique_TP": unique_result[0],
            "unique_FP": unique_result[1],
            "unique_FN": unique_result[2],
            "unique_TN": unique_result[3],
            "unique_precision": unique_result[4],
            "unique_recall": unique_result[5],
            "unique_f1": unique_result[6],
            "correspondence_count_mismatches": count_mismatches,
            "reference_inlier_count_mismatches": inlier_mismatches
        })

    return output_rows


def main():
    if not METRICS_FILE.is_file():
        raise FileNotFoundError(METRICS_FILE)

    metrics = load_metrics()

    print("=" * 100)
    print("AQUALOC SEQUENCE 06 - CORRESPONDENCE UNIQUENESS AUDIT")
    print("=" * 100)

    print("Metric records:", len(metrics))
    print("Expected:", 86190)

    if len(metrics) != 86190:
        raise ValueError("Metric row count mismatch")

    results = []

    for detector, preprocessing in CONFIGS:
        print(f"\n[CHECK] {detector} + {preprocessing}")

        rows = audit_configuration(
            detector,
            preprocessing,
            metrics
        )

        results.extend(rows)

        for r in rows:
            print(
                f"{r['scope']}: "
                f"collision={r['inlier_collision_percentage']:.2f}%, "
                f"raw F1={r['raw_f1']:.4f}, "
                f"unique-match F1={r['unique_f1']:.4f}, "
                f"count mismatches={r['correspondence_count_mismatches']}, "
                f"inlier mismatches={r['reference_inlier_count_mismatches']}"
            )

    output = RESULT_DIR / "sequence6_correspondence_uniqueness_audit.csv"

    with output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)

    print("\nOUTPUT:")
    print(output)
    print("\nOriginal benchmark modified: NO")
    print("Calibrated reference modified: NO")
    print("=" * 100)


if __name__ == "__main__":
    main()
