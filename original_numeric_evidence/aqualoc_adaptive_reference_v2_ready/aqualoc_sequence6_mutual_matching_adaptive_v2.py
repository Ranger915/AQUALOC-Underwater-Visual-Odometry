import csv
import gzip
import json
import math
import sys
import time
from pathlib import Path
from datetime import datetime, timezone

import cv2
import numpy as np
from adaptive_clahe import AdaptiveConfig, apply_adaptive

ADAPTIVE_CONFIG = AdaptiveConfig(version="v2")

ROOT = Path.home() / "Downloads" / "archaeo_sequence_6_raw_data"

IMAGE_DIR = ROOT / "raw_data" / "images_sequence_6"
REF_DIR = ROOT / "raw_data" / "sequence6_calibrated_reference"

PAIR_FILE = REF_DIR / "sequence6_calibrated_pair_manifest.csv"

OUT = ROOT / "sequence6_adaptive_v2_mutual_matching_results"
OUT.mkdir(parents=True, exist_ok=True)

K = np.array([
    [543.3327734182214, 0.0, 489.02536042247897],
    [0.0, 542.398772982566, 305.38727712002805],
    [0.0, 0.0, 1.0]
], dtype=np.float64)

D = np.array([
    -0.1255945656257394,
     0.053221287232781606,
     9.94070021080493e-05,
     9.550660927242349e-05
], dtype=np.float64)

WIDTH = 968
HEIGHT = 608

MAX_FEATURES = 1500
RATIO_TEST = 0.75

ERROR_THRESHOLDS = [1.0, 2.0, 3.0, 5.0]
MINIMUM_SUPPORT = [8, 12, 20]


def make_detector(name):
    if name == "ORB":
        return cv2.ORB_create(nfeatures=MAX_FEATURES), cv2.NORM_HAMMING

    if name == "AKAZE_MLDB":
        return cv2.AKAZE_create(), cv2.NORM_HAMMING

    if name == "SIFT":
        if not hasattr(cv2, "SIFT_create"):
            raise RuntimeError("SIFT is unavailable in this OpenCV build.")
        return cv2.SIFT_create(nfeatures=MAX_FEATURES), cv2.NORM_L2

    raise ValueError(name)


def read_pairs():
    pairs = []

    with PAIR_FILE.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)

        for row in reader:
            F = np.array([
                [float(row[f"F_{r}{c}"]) for c in range(3)]
                for r in range(3)
            ], dtype=np.float64)

            if not np.all(np.isfinite(F)) or np.linalg.norm(F) < 1e-12:
                raise ValueError("Invalid reference fundamental matrix")

            pairs.append({
                "i": int(row["frame_id_i"]),
                "j": int(row["frame_id_j"]),
                "gap": int(row["frame_gap"]),
                "label": int(row["colmap_overlap_label"]),
                "baseline": float(row["baseline_norm_export_units"]),
                "rotation": float(row["relative_rotation_degrees"]),
                "F": F
            })

    if len(pairs) != 14365:
        raise ValueError(f"Expected 14365 pairs, got {len(pairs)}")

    return pairs


def extract_features(frame_ids, detector_name, preprocessing):
    detector, norm = make_detector(detector_name)

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    feature_map = {}
    extraction_total = 0.0

    print(f"\n[EXTRACT] {detector_name} + {preprocessing}", flush=True)

    for index, frame_id in enumerate(frame_ids, 1):
        path = IMAGE_DIR / f"frame{frame_id:06d}.png"
        gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)

        if gray is None:
            raise RuntimeError(f"Cannot read {path}")

        if gray.shape != (HEIGHT, WIDTH):
            raise ValueError(f"Unexpected image dimensions: {path.name}")

        if preprocessing == "adaptive_v2":
            image, _ = apply_adaptive(gray, ADAPTIVE_CONFIG)
        else:
            image = clahe.apply(gray) if preprocessing == "clahe" else gray

        t0 = time.perf_counter()
        keypoints, descriptors = detector.detectAndCompute(image, None)
        elapsed = time.perf_counter() - t0
        extraction_total += elapsed

        if descriptors is not None and len(keypoints) > MAX_FEATURES:
            indices = np.argsort(
                [-kp.response for kp in keypoints]
            )[:MAX_FEATURES]

            keypoints = [keypoints[k] for k in indices]
            descriptors = descriptors[indices]

        raw_xy = np.asarray(
            [kp.pt for kp in keypoints],
            dtype=np.float32
        ).reshape(-1, 2)

        if len(raw_xy):
            und_xy = cv2.undistortPoints(
                raw_xy.reshape(-1, 1, 2),
                K,
                D,
                P=K
            ).reshape(-1, 2)
        else:
            und_xy = np.empty((0, 2), dtype=np.float32)

        feature_map[frame_id] = {
            "keypoints": keypoints,
            "descriptors": descriptors,
            "raw_xy": raw_xy,
            "und_xy": und_xy
        }

        if index % 25 == 0 or index == len(frame_ids):
            print(f"  extracted {index}/{len(frame_ids)}", flush=True)

    return feature_map, norm, extraction_total


def ratio_filter(knn):
    accepted = {}

    for candidates in knn:
        if len(candidates) < 2:
            continue

        first, second = candidates[0], candidates[1]

        if first.distance < RATIO_TEST * second.distance:
            accepted[first.queryIdx] = first

    return accepted


def mutual_ratio_matches(matcher, des1, des2):
    """
    A match is accepted only when:
      1. It passes the forward Lowe ratio test.
      2. It passes the reverse Lowe ratio test.
      3. Both directions select the same keypoint pair.

    This produces one-to-one assignments among accepted matches.
    """

    forward_knn = matcher.knnMatch(des1, des2, k=2)
    reverse_knn = matcher.knnMatch(des2, des1, k=2)

    forward = ratio_filter(forward_knn)
    reverse = ratio_filter(reverse_knn)

    mutual = []

    for query_index, match in forward.items():
        reverse_match = reverse.get(match.trainIdx)

        if (
            reverse_match is not None
            and reverse_match.trainIdx == query_index
        ):
            mutual.append(match)

    return forward, reverse, mutual


def sampson_distance(p1, p2, F):
    x1 = np.column_stack([p1, np.ones(len(p1))])
    x2 = np.column_stack([p2, np.ones(len(p2))])

    Fx1 = (F @ x1.T).T
    Ftx2 = (F.T @ x2.T).T

    residual = np.sum(x2 * Fx1, axis=1)

    denominator = (
        Fx1[:, 0] ** 2
        + Fx1[:, 1] ** 2
        + Ftx2[:, 0] ** 2
        + Ftx2[:, 1] ** 2
    )

    result = np.full(len(p1), np.inf, dtype=float)
    valid = denominator > 1e-12

    result[valid] = np.sqrt(
        residual[valid] ** 2 / denominator[valid]
    )

    return result


def average_precision(labels, scores):
    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=float)

    positive_count = int(labels.sum())

    if positive_count == 0:
        return float("nan")

    order = np.argsort(-scores, kind="stable")
    labels = labels[order]
    scores = scores[order]

    tp = 0
    fp = 0
    previous_recall = 0.0
    ap = 0.0
    index = 0

    while index < len(labels):
        score = scores[index]
        end = index

        while end < len(labels) and scores[end] == score:
            end += 1

        group = labels[index:end]
        group_tp = int(group.sum())

        tp += group_tp
        fp += len(group) - group_tp

        recall = tp / positive_count
        precision = tp / (tp + fp)

        ap += (recall - previous_recall) * precision
        previous_recall = recall
        index = end

    return float(ap)


def confusion(labels, predictions):
    tp = sum(y == 1 and p == 1 for y, p in zip(labels, predictions))
    fp = sum(y == 0 and p == 1 for y, p in zip(labels, predictions))
    fn = sum(y == 1 and p == 0 for y, p in zip(labels, predictions))
    tn = sum(y == 0 and p == 0 for y, p in zip(labels, predictions))

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall else 0.0
    )

    return tp, fp, fn, tn, precision, recall, f1


def process_configuration(detector_name, preprocessing, pairs, frame_ids):
    feature_map, norm, extraction_time = extract_features(
        frame_ids,
        detector_name,
        preprocessing
    )

    matcher = cv2.BFMatcher(norm)

    config = f"{detector_name}_{preprocessing}"
    detail_path = OUT / f"mutual_matches_{config}.csv.gz"

    metric_rows = []

    detail_header = [
        "frame_id_i", "frame_id_j", "frame_gap", "overlap_label",
        "query_keypoint_index", "train_keypoint_index",
        "descriptor_distance",
        "raw_x_i", "raw_y_i", "raw_x_j", "raw_y_j",
        "undistorted_x_i", "undistorted_y_i",
        "undistorted_x_j", "undistorted_y_j",
        "sampson_distance_px"
    ]

    with gzip.open(detail_path, "wt", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(detail_header)

        for pair_index, pair in enumerate(pairs, 1):
            f1 = feature_map[pair["i"]]
            f2 = feature_map[pair["j"]]

            des1 = f1["descriptors"]
            des2 = f2["descriptors"]

            row = {
                "detector": detector_name,
                "preprocessing": preprocessing,
                "frame_id_i": pair["i"],
                "frame_id_j": pair["j"],
                "frame_gap": pair["gap"],
                "overlap_label": pair["label"],
                "baseline_export_units": pair["baseline"],
                "keypoints_i": len(f1["keypoints"]),
                "keypoints_j": len(f2["keypoints"]),
                "forward_ratio_matches": 0,
                "reverse_ratio_matches": 0,
                "mutual_matches": 0,
                "mutual_inliers_1px": 0,
                "mutual_inliers_2px": 0,
                "mutual_inliers_3px": 0,
                "mutual_inliers_5px": 0,
                "mutual_fraction_1px": 0.0,
                "mutual_fraction_2px": 0.0,
                "mutual_fraction_3px": 0.0,
                "mutual_fraction_5px": 0.0,
                "mean_sampson_px": "",
                "median_sampson_px": "",
                "matching_time_s": 0.0,
                "status": "ok"
            }

            if (
                des1 is None or des2 is None
                or len(des1) < 2 or len(des2) < 2
            ):
                row["status"] = "insufficient_descriptors"
                metric_rows.append(row)
                continue

            t0 = time.perf_counter()

            forward, reverse, mutual = mutual_ratio_matches(
                matcher, des1, des2
            )

            row["matching_time_s"] = time.perf_counter() - t0
            row["forward_ratio_matches"] = len(forward)
            row["reverse_ratio_matches"] = len(reverse)
            row["mutual_matches"] = len(mutual)

            # Enforce and verify the one-to-one assignment property.
            query_ids = [m.queryIdx for m in mutual]
            train_ids = [m.trainIdx for m in mutual]

            if (
                len(query_ids) != len(set(query_ids))
                or len(train_ids) != len(set(train_ids))
            ):
                raise RuntimeError(
                    f"Non-unique mutual correspondence in pair "
                    f"{pair['i']}, {pair['j']}"
                )

            if not mutual:
                row["status"] = "no_mutual_matches"
                metric_rows.append(row)
                continue

            qi = np.asarray(query_ids, dtype=int)
            ti = np.asarray(train_ids, dtype=int)

            raw1 = f1["raw_xy"][qi]
            raw2 = f2["raw_xy"][ti]
            und1 = f1["und_xy"][qi]
            und2 = f2["und_xy"][ti]

            distances = sampson_distance(
                und1, und2, pair["F"]
            )

            valid = np.isfinite(distances)

            if np.any(valid):
                row["mean_sampson_px"] = float(
                    np.mean(distances[valid])
                )
                row["median_sampson_px"] = float(
                    np.median(distances[valid])
                )

            for threshold in ERROR_THRESHOLDS:
                count = int(np.sum(distances <= threshold))
                suffix = f"{int(threshold)}px"

                row[f"mutual_inliers_{suffix}"] = count
                row[f"mutual_fraction_{suffix}"] = (
                    count / len(mutual)
                )

            for index, match in enumerate(mutual):
                writer.writerow([
                    pair["i"],
                    pair["j"],
                    pair["gap"],
                    pair["label"],
                    match.queryIdx,
                    match.trainIdx,
                    match.distance,
                    raw1[index, 0],
                    raw1[index, 1],
                    raw2[index, 0],
                    raw2[index, 1],
                    und1[index, 0],
                    und1[index, 1],
                    und2[index, 0],
                    und2[index, 1],
                    distances[index]
                ])

            metric_rows.append(row)

            if pair_index % 1000 == 0 or pair_index == len(pairs):
                print(
                    f"  matched {pair_index}/{len(pairs)} pairs",
                    flush=True
                )

    return metric_rows, extraction_time, detail_path


def make_summary(rows):
    summaries = []

    scopes = [
        ("all_pairs", lambda r: True),
        ("local_gap_le_100", lambda r: r["frame_gap"] <= 100),
        ("nonlocal_gap_ge_200", lambda r: r["frame_gap"] >= 200)
    ]

    for detector in ("ORB", "AKAZE_MLDB", "SIFT"):
        for preprocessing in ("adaptive_v2",):

            config = [
                r for r in rows
                if r["detector"] == detector
                and r["preprocessing"] == preprocessing
            ]

            for scope, condition in scopes:
                selected = [r for r in config if condition(r)]

                labels = [
                    int(r["overlap_label"]) for r in selected
                ]

                for threshold in ERROR_THRESHOLDS:
                    suffix = f"{int(threshold)}px"
                    count_key = f"mutual_inliers_{suffix}"
                    fraction_key = f"mutual_fraction_{suffix}"

                    counts = [
                        int(r[count_key]) for r in selected
                    ]

                    fractions = [
                        float(r[fraction_key]) for r in selected
                    ]

                    ap_count = average_precision(labels, counts)
                    ap_fraction = average_precision(labels, fractions)

                    for minimum in MINIMUM_SUPPORT:
                        predictions = [
                            int(r[count_key] >= minimum)
                            for r in selected
                        ]

                        (
                            tp, fp, fn, tn,
                            precision, recall, f1
                        ) = confusion(labels, predictions)

                        summaries.append({
                            "detector": detector,
                            "preprocessing": preprocessing,
                            "scope": scope,
                            "sampson_threshold_px": threshold,
                            "minimum_mutual_inliers": minimum,
                            "pair_count": len(selected),
                            "positive_pairs": sum(labels),
                            "negative_pairs": len(labels) - sum(labels),
                            "TP": tp,
                            "FP": fp,
                            "FN": fn,
                            "TN": tn,
                            "precision": precision,
                            "recall": recall,
                            "f1": f1,
                            "average_precision_count": ap_count,
                            "average_precision_fraction": ap_fraction
                        })

    return summaries


def main():
    # Do not silently overwrite a previous full adaptive experiment.
    if (OUT / "sequence6_mutual_feature_metrics.csv").exists():
        raise FileExistsError(
            "Adaptive v2 results already exist in " + str(OUT) +
            ". Move or archive the directory before rerunning."
        )
    print("=" * 84)
    print("AQUALOC SEQUENCE 06 - BIDIRECTIONAL MUTUAL MATCHING BENCHMARK")
    print("=" * 84)

    if not IMAGE_DIR.is_dir():
        raise FileNotFoundError(IMAGE_DIR)

    pairs = read_pairs()

    frame_ids = sorted(
        {p["i"] for p in pairs} | {p["j"] for p in pairs}
    )

    if len(frame_ids) != 170:
        raise ValueError(f"Expected 170 images, found {len(frame_ids)}")

    print("Images:", len(frame_ids))
    print("Pairs:", len(pairs))
    print("Feature budget:", MAX_FEATURES)
    print("Lowe ratio:", RATIO_TEST)
    print("Mutual ratio test: enabled")
    print("Reference Sampson thresholds:", ERROR_THRESHOLDS)

    all_rows = []
    extraction_summary = []

    for preprocessing in ("adaptive_v2",):
        for detector in ("ORB", "AKAZE_MLDB", "SIFT"):

            rows, extraction_time, detail_path = process_configuration(
                detector,
                preprocessing,
                pairs,
                frame_ids
            )

            all_rows.extend(rows)

            extraction_summary.append({
                "detector": detector,
                "preprocessing": preprocessing,
                "extraction_total_s": extraction_time,
                "extraction_mean_ms_per_image": (
                    extraction_time / len(frame_ids) * 1000
                ),
                "correspondence_file": str(detail_path),
                "correspondence_file_size_bytes": detail_path.stat().st_size
            })

    expected = len(pairs) * 3

    if len(all_rows) != expected:
        raise RuntimeError(
            f"Expected {expected} metric rows; got {len(all_rows)}"
        )

    metrics_path = OUT / "sequence6_mutual_feature_metrics.csv"

    with metrics_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(all_rows[0].keys())
        )
        writer.writeheader()
        writer.writerows(all_rows)

    summaries = make_summary(all_rows)

    summary_path = OUT / "sequence6_mutual_feature_summary.csv"

    with summary_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(summaries[0].keys())
        )
        writer.writeheader()
        writer.writerows(summaries)

    metadata = {
        "dataset": "AQUALOC Sequence 06",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "opencv_version": cv2.__version__,
        "numpy_version": np.__version__,
        "image_count": len(frame_ids),
        "pair_count": len(pairs),
        "metric_rows": len(all_rows),
        "expected_metric_rows": expected,
        "detectors": ["ORB", "AKAZE_MLDB", "SIFT"],
        "preprocessing": ["adaptive_v2"],
        "adaptive_config": ADAPTIVE_CONFIG.__dict__,
        "decision_rule": "RAW if mean_local_std >= 10.5; otherwise CLAHE clip 2.0 or 3.0 based on original severe thresholds",
        "feature_budget": MAX_FEATURES,
        "ratio_threshold": RATIO_TEST,
        "matching": (
            "Bidirectional Lowe ratio test; retain a match only if "
            "the best ratio-filtered correspondence agrees in both directions"
        ),
        "one_to_one_property": (
            "Each accepted source and target keypoint appears at most once"
        ),
        "matcher_norms": {
            "ORB": "Hamming",
            "AKAZE_MLDB": "Hamming",
            "SIFT": "L2"
        },
        "camera_model": "Pinhole RadTan",
        "camera_matrix": K.tolist(),
        "distortion_coefficients": D.tolist(),
        "point_undistortion": "cv2.undistortPoints with P=K",
        "reference_geometry": "Fixed COLMAP-pose-derived F",
        "geometric_error": "Sampson distance in undistorted pixel coordinates",
        "error_thresholds_px": ERROR_THRESHOLDS,
        "original_pair_set": "All 14,365 reference pairs",
        "original_benchmark_modified": False,
        "reference_caveat": (
            "COLMAP-derived comparative reference, not absolute physical "
            "ground truth; original reconstruction used exhaustive SIFT matching"
        ),
        "outputs": {
            "metrics": str(metrics_path),
            "summary": str(summary_path)
        },
        "extraction": extraction_summary
    }

    metadata_path = OUT / "sequence6_mutual_feature_metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, indent=4),
        encoding="utf-8"
    )

    print("\n" + "=" * 84)
    print("MUTUAL MATCHING BENCHMARK COMPLETE")
    print("=" * 84)
    print("Metric rows:", len(all_rows))
    print("Expected rows:", expected)

    print("\nMAIN OUTPUTS")
    print(metrics_path)
    print(summary_path)
    print(metadata_path)

    print("\nDETAIL FILES")
    for item in extraction_summary:
        print(item["correspondence_file"])
        print("Size:", item["correspondence_file_size_bytes"])

    print("\nOriginal 60,774-row benchmark modified: NO")
    print("Original calibrated feature results modified: NO")
    print("=" * 84)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"\n[ERROR] {exc}", file=sys.stderr)
        sys.exit(1)
