import csv
import gzip
import json
import re
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

OUT = ROOT / "sequence6_adaptive_v2_calibrated_reference_results"
OUT.mkdir(parents=True, exist_ok=True)

# Official AQUALOC camera calibration
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

SAMPSON_THRESHOLDS = [1.0, 2.0, 3.0, 5.0]
MIN_INLIERS = [8, 12, 20]

NONLOCAL_GAP = 200
LOCAL_MAX_GAP = 100


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


def prepare_image(gray, preprocessing, clahe):
    if preprocessing == "raw":
        return gray

    if preprocessing == "clahe":
        return clahe.apply(gray)

    if preprocessing == "adaptive_v2":
        processed, _ = apply_adaptive(gray, ADAPTIVE_CONFIG)
        return processed

    raise ValueError(preprocessing)


def extract_features(frame_ids, detector_name, preprocessing):
    detector, norm = make_detector(detector_name)

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    features = {}
    total_extraction_time = 0.0

    print(
        f"\n[EXTRACT] {preprocessing.upper()} + {detector_name}",
        flush=True
    )

    for index, frame_id in enumerate(frame_ids, 1):

        path = IMAGE_DIR / f"frame{frame_id:06d}.png"

        if not path.is_file():
            raise FileNotFoundError(path)

        gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)

        if gray is None:
            raise RuntimeError(f"Could not read image: {path}")

        if gray.shape != (HEIGHT, WIDTH):
            raise ValueError(
                f"Unexpected dimensions in {path.name}: {gray.shape}"
            )

        image = prepare_image(gray, preprocessing, clahe)

        t0 = time.perf_counter()
        keypoints, descriptors = detector.detectAndCompute(image, None)
        elapsed = time.perf_counter() - t0

        # Preserve the original benchmark's 1500-feature ceiling.
        if descriptors is not None and len(keypoints) > MAX_FEATURES:
            order = np.argsort(
                [-kp.response for kp in keypoints]
            )[:MAX_FEATURES]

            keypoints = [keypoints[i] for i in order]
            descriptors = descriptors[order]

        raw_xy = np.asarray(
            [kp.pt for kp in keypoints],
            dtype=np.float32
        ).reshape(-1, 2)

        # Correct the detected feature coordinates using official RadTan
        # calibration. P=K retains undistorted pixel coordinates.
        if len(raw_xy):
            und_xy = cv2.undistortPoints(
                raw_xy.reshape(-1, 1, 2),
                K,
                D,
                P=K
            ).reshape(-1, 2)
        else:
            und_xy = np.empty((0, 2), dtype=np.float32)

        features[frame_id] = {
            "keypoints": keypoints,
            "descriptors": descriptors,
            "raw_xy": raw_xy,
            "und_xy": und_xy,
            "extract_time_s": elapsed
        }

        total_extraction_time += elapsed

        if index % 25 == 0 or index == len(frame_ids):
            print(
                f"  extracted {index}/{len(frame_ids)} images",
                flush=True
            )

    return features, norm, total_extraction_time


def read_pair_manifest():
    if not PAIR_FILE.is_file():
        raise FileNotFoundError(PAIR_FILE)

    pairs = []

    with PAIR_FILE.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)

        required = [
            "frame_id_i",
            "frame_id_j",
            "frame_gap",
            "colmap_overlap_label",
            "baseline_norm_export_units"
        ]

        for column in required:
            if column not in (reader.fieldnames or []):
                raise ValueError(f"Missing manifest column: {column}")

        for row in reader:
            i = int(row["frame_id_i"])
            j = int(row["frame_id_j"])

            F = np.array([
                [
                    float(row[f"F_{r}{c}"])
                    for c in range(3)
                ]
                for r in range(3)
            ], dtype=np.float64)

            if not np.all(np.isfinite(F)):
                raise ValueError(f"Invalid F matrix for pair {i}, {j}")

            if np.linalg.norm(F) < 1e-12:
                raise ValueError(f"Degenerate F matrix for pair {i}, {j}")

            pairs.append({
                "frame_id_i": i,
                "frame_id_j": j,
                "frame_gap": int(row["frame_gap"]),
                "overlap_label": int(row["colmap_overlap_label"]),
                "baseline": float(row["baseline_norm_export_units"]),
                "rotation_degrees": float(
                    row["relative_rotation_degrees"]
                ),
                "F": F
            })

    expected = 170 * 169 // 2

    if len(pairs) != expected:
        raise ValueError(
            f"Expected {expected} pairs, found {len(pairs)}"
        )

    return pairs


def sampson_distance(points1, points2, F):
    """
    Sampson distance in undistorted pixel coordinates.

    x2.T @ F @ x1 = 0
    """

    x1 = np.column_stack([
        points1,
        np.ones(len(points1))
    ])

    x2 = np.column_stack([
        points2,
        np.ones(len(points2))
    ])

    Fx1 = (F @ x1.T).T
    Ftx2 = (F.T @ x2.T).T

    residual = np.sum(x2 * Fx1, axis=1)

    denominator = (
        Fx1[:, 0] ** 2
        + Fx1[:, 1] ** 2
        + Ftx2[:, 0] ** 2
        + Ftx2[:, 1] ** 2
    )

    distances = np.full(len(points1), np.inf, dtype=np.float64)

    valid = denominator > 1e-12

    distances[valid] = np.sqrt(
        residual[valid] ** 2 / denominator[valid]
    )

    return distances


def grid_coverage(points, grid=4):
    if len(points) == 0:
        return 0.0

    xs = np.clip(
        (points[:, 0] / WIDTH * grid).astype(int),
        0,
        grid - 1
    )

    ys = np.clip(
        (points[:, 1] / HEIGHT * grid).astype(int),
        0,
        grid - 1
    )

    return len(set(zip(xs.tolist(), ys.tolist()))) / float(grid * grid)


def average_precision(labels, scores):
    """
    Average precision with equal-score groups processed together.
    """

    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=float)

    positives = int(labels.sum())

    if positives == 0:
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

        tp += int(group.sum())
        fp += len(group) - int(group.sum())

        recall = tp / positives
        precision = tp / max(tp + fp, 1)

        ap += (recall - previous_recall) * precision
        previous_recall = recall

        index = end

    return float(ap)


def confusion_metrics(labels, predictions):
    labels = np.asarray(labels, dtype=int)
    predictions = np.asarray(predictions, dtype=int)

    tp = int(np.sum((labels == 1) & (predictions == 1)))
    fp = int(np.sum((labels == 0) & (predictions == 1)))
    fn = int(np.sum((labels == 1) & (predictions == 0)))
    tn = int(np.sum((labels == 0) & (predictions == 0)))

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0

    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall else 0.0
    )

    return tp, fp, fn, tn, precision, recall, f1


def run_configuration(detector_name, preprocessing, pairs, frame_ids):
    features, norm, extraction_time = extract_features(
        frame_ids,
        detector_name,
        preprocessing
    )

    matcher = cv2.BFMatcher(norm)

    config_name = f"{detector_name}_{preprocessing}"

    detail_path = OUT / f"matches_{config_name}.csv.gz"

    metrics = []

    detail_header = [
        "frame_id_i",
        "frame_id_j",
        "frame_gap",
        "overlap_label",
        "baseline_export_units",
        "query_keypoint_index",
        "train_keypoint_index",
        "descriptor_distance",
        "raw_x_i",
        "raw_y_i",
        "raw_x_j",
        "raw_y_j",
        "undistorted_x_i",
        "undistorted_y_i",
        "undistorted_x_j",
        "undistorted_y_j",
        "sampson_distance_px"
    ]

    print(f"\n[MATCH] {config_name}", flush=True)

    with gzip.open(
        detail_path,
        "wt",
        newline="",
        encoding="utf-8"
    ) as detail_file:

        detail_writer = csv.writer(detail_file)
        detail_writer.writerow(detail_header)

        for pair_index, pair in enumerate(pairs, 1):

            fid1 = pair["frame_id_i"]
            fid2 = pair["frame_id_j"]

            f1 = features[fid1]
            f2 = features[fid2]

            des1 = f1["descriptors"]
            des2 = f2["descriptors"]

            record = {
                "detector": detector_name,
                "preprocessing": preprocessing,
                "frame_id_i": fid1,
                "frame_id_j": fid2,
                "frame_gap": pair["frame_gap"],
                "overlap_label": pair["overlap_label"],
                "baseline_export_units": pair["baseline"],
                "rotation_degrees": pair["rotation_degrees"],
                "keypoints_i": len(f1["keypoints"]),
                "keypoints_j": len(f2["keypoints"]),
                "raw_knn_matches": 0,
                "ratio_matches": 0,
                "matching_time_s": 0.0,
                "geometry_time_s": 0.0,
                "reference_inliers_1px": 0,
                "reference_inliers_2px": 0,
                "reference_inliers_3px": 0,
                "reference_inliers_5px": 0,
                "reference_inlier_ratio_1px": 0.0,
                "reference_inlier_ratio_2px": 0.0,
                "reference_inlier_ratio_3px": 0.0,
                "reference_inlier_ratio_5px": 0.0,
                "mean_sampson_px": "",
                "median_sampson_px": "",
                "p90_sampson_px": "",
                "coverage_i_2px": 0.0,
                "coverage_j_2px": 0.0,
                "status": "ok"
            }

            if (
                des1 is None or des2 is None
                or len(des1) < 2 or len(des2) < 2
            ):
                record["status"] = "insufficient_descriptors"
                metrics.append(record)
                continue

            t0 = time.perf_counter()

            knn = matcher.knnMatch(des1, des2, k=2)

            good = [
                m for candidates in knn
                if len(candidates) >= 2
                for m, n in [candidates[:2]]
                if m.distance < RATIO_TEST * n.distance
            ]

            record["matching_time_s"] = time.perf_counter() - t0
            record["raw_knn_matches"] = len(knn)
            record["ratio_matches"] = len(good)

            if not good:
                record["status"] = "no_ratio_matches"
                metrics.append(record)
                continue

            query_indices = np.asarray(
                [m.queryIdx for m in good],
                dtype=int
            )

            train_indices = np.asarray(
                [m.trainIdx for m in good],
                dtype=int
            )

            raw1 = f1["raw_xy"][query_indices]
            raw2 = f2["raw_xy"][train_indices]

            und1 = f1["und_xy"][query_indices]
            und2 = f2["und_xy"][train_indices]

            t0 = time.perf_counter()

            distances = sampson_distance(
                und1,
                und2,
                pair["F"]
            )

            finite = np.isfinite(distances)
            valid_distances = distances[finite]

            if len(valid_distances):
                record["mean_sampson_px"] = float(
                    np.mean(valid_distances)
                )
                record["median_sampson_px"] = float(
                    np.median(valid_distances)
                )
                record["p90_sampson_px"] = float(
                    np.percentile(valid_distances, 90)
                )

            for threshold in SAMPSON_THRESHOLDS:
                count = int(np.sum(distances <= threshold))
                key = f"reference_inliers_{int(threshold)}px"
                ratio_key = f"reference_inlier_ratio_{int(threshold)}px"

                record[key] = count
                record[ratio_key] = count / len(good)

            accepted = distances <= 2.0

            record["coverage_i_2px"] = grid_coverage(raw1[accepted])
            record["coverage_j_2px"] = grid_coverage(raw2[accepted])

            record["geometry_time_s"] = time.perf_counter() - t0

            for k, match in enumerate(good):
                detail_writer.writerow([
                    fid1,
                    fid2,
                    pair["frame_gap"],
                    pair["overlap_label"],
                    pair["baseline"],
                    match.queryIdx,
                    match.trainIdx,
                    match.distance,
                    raw1[k, 0],
                    raw1[k, 1],
                    raw2[k, 0],
                    raw2[k, 1],
                    und1[k, 0],
                    und1[k, 1],
                    und2[k, 0],
                    und2[k, 1],
                    distances[k]
                ])

            metrics.append(record)

            if pair_index % 1000 == 0 or pair_index == len(pairs):
                print(
                    f"  processed {pair_index}/{len(pairs)} pairs",
                    flush=True
                )

    return metrics, extraction_time, detail_path


def build_summary(all_metrics):
    summary_rows = []

    scopes = [
        ("all_pairs", lambda r: True),
        ("local_gap_le_100", lambda r: r["frame_gap"] <= LOCAL_MAX_GAP),
        ("nonlocal_gap_ge_200", lambda r: r["frame_gap"] >= NONLOCAL_GAP)
    ]

    for detector in ("ORB", "AKAZE_MLDB", "SIFT"):
        for preprocessing in ("adaptive_v2",):

            config_rows = [
                r for r in all_metrics
                if r["detector"] == detector
                and r["preprocessing"] == preprocessing
            ]

            for scope_name, condition in scopes:

                selected = [
                    r for r in config_rows if condition(r)
                ]

                if not selected:
                    continue

                labels = [
                    r["overlap_label"] for r in selected
                ]

                positive_count = sum(labels)
                negative_count = len(labels) - positive_count

                scores_2px = [
                    r["reference_inliers_2px"] for r in selected
                ]

                ap = average_precision(labels, scores_2px)

                correspondence_total = sum(
                    r["ratio_matches"] for r in selected
                )

                for threshold in SAMPSON_THRESHOLDS:

                    count_key = (
                        f"reference_inliers_{int(threshold)}px"
                    )

                    total_inliers = sum(
                        r[count_key] for r in selected
                    )

                    correspondence_fraction = (
                        total_inliers / correspondence_total
                        if correspondence_total else 0.0
                    )

                    for minimum in MIN_INLIERS:

                        predictions = [
                            int(r[count_key] >= minimum)
                            for r in selected
                        ]

                        (
                            tp, fp, fn, tn,
                            precision, recall, f1
                        ) = confusion_metrics(labels, predictions)

                        summary_rows.append({
                            "detector": detector,
                            "preprocessing": preprocessing,
                            "scope": scope_name,
                            "metric": "overlap_detection",
                            "sampson_threshold_px": threshold,
                            "minimum_reference_inliers": minimum,
                            "pair_count": len(selected),
                            "positive_pairs": positive_count,
                            "negative_pairs": negative_count,
                            "true_positive": tp,
                            "false_positive": fp,
                            "false_negative": fn,
                            "true_negative": tn,
                            "precision": precision,
                            "recall": recall,
                            "f1": f1,
                            "average_precision_2px": ap,
                            "correspondence_epipolar_fraction": correspondence_fraction
                        })

    return summary_rows


def main():
    # Do not silently overwrite a previous full adaptive experiment.
    if (OUT / "sequence6_reference_feature_metrics.csv").exists():
        raise FileExistsError(
            "Adaptive v2 results already exist in " + str(OUT) +
            ". Move or archive the directory before rerunning."
        )
    print("=" * 76)
    print("AQUALOC SEQUENCE 06 - CALIBRATED FEATURE BENCHMARK")
    print("=" * 76)

    if not IMAGE_DIR.is_dir():
        raise FileNotFoundError(IMAGE_DIR)

    pairs = read_pair_manifest()

    frame_ids = sorted({
        p["frame_id_i"] for p in pairs
    } | {
        p["frame_id_j"] for p in pairs
    })

    if len(frame_ids) != 170:
        raise ValueError(
            f"Expected 170 unique reference frames, found {len(frame_ids)}"
        )

    print("\nINPUT")
    print("-" * 76)
    print("Image directory:", IMAGE_DIR)
    print("Reference manifest:", PAIR_FILE)
    print("Unique frames:", len(frame_ids))
    print("Image pairs:", len(pairs))
    print("OpenCV version:", cv2.__version__)
    print("NumPy version:", np.__version__)
    print("Feature budget:", MAX_FEATURES)
    print("Lowe ratio:", RATIO_TEST)
    print("Sampson thresholds:", SAMPSON_THRESHOLDS)

    all_metrics = []
    extraction_summary = []

    for preprocessing in ("adaptive_v2",):
        for detector in ("ORB", "AKAZE_MLDB", "SIFT"):

            metrics, extraction_time, detail_path = run_configuration(
                detector,
                preprocessing,
                pairs,
                frame_ids
            )

            all_metrics.extend(metrics)

            extraction_summary.append({
                "detector": detector,
                "preprocessing": preprocessing,
                "extraction_time_total_s": extraction_time,
                "mean_extraction_time_per_frame_s": (
                    extraction_time / len(frame_ids)
                ),
                "correspondence_file": str(detail_path),
                "correspondence_file_size_bytes": detail_path.stat().st_size
            })

    metrics_path = OUT / "sequence6_reference_feature_metrics.csv"

    if not all_metrics:
        raise RuntimeError("No feature metrics were generated.")

    metric_fields = list(all_metrics[0].keys())

    with metrics_path.open(
        "w", newline="", encoding="utf-8"
    ) as f:
        writer = csv.DictWriter(f, fieldnames=metric_fields)
        writer.writeheader()
        writer.writerows(all_metrics)

    summary_rows = build_summary(all_metrics)

    summary_path = OUT / "sequence6_reference_feature_summary.csv"

    with summary_path.open(
        "w", newline="", encoding="utf-8"
    ) as f:
        fields = list(summary_rows[0].keys())
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summary_rows)

    metadata = {
        "dataset": "AQUALOC Sequence 06",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "opencv_version": cv2.__version__,
        "numpy_version": np.__version__,
        "image_count": len(frame_ids),
        "pair_count": len(pairs),
        "expected_metric_rows": len(pairs) * 3,
        "actual_metric_rows": len(all_metrics),
        "detectors": ["ORB", "AKAZE_MLDB", "SIFT"],
        "preprocessing": ["adaptive_v2"],
        "adaptive_config": ADAPTIVE_CONFIG.__dict__,
        "decision_rule": "RAW if mean_local_std >= 10.5; otherwise CLAHE clip 2.0 or 3.0 based on original severe thresholds",
        "max_features": MAX_FEATURES,
        "ratio_test": RATIO_TEST,
        "matcher_norms": {
            "ORB": "Hamming",
            "AKAZE_MLDB": "Hamming",
            "SIFT": "L2"
        },
        "camera_matrix": K.tolist(),
        "distortion_model": "RadTan",
        "distortion_coefficients": D.tolist(),
        "undistortion": "cv2.undistortPoints with P=K",
        "reference_geometry": (
            "Fixed COLMAP-pose-derived fundamental matrix from pair manifest"
        ),
        "error_metric": (
            "Sampson distance in undistorted pixel coordinates"
        ),
        "sampson_thresholds_px": SAMPSON_THRESHOLDS,
        "pair_sampling": "All 14,365 pairs; no negative downsampling",
        "loop_evaluation_subset": (
            f"frame_gap >= {NONLOCAL_GAP}; "
            "this excludes temporal neighbours by a declared fixed threshold"
        ),
        "local_subset": f"frame_gap <= {LOCAL_MAX_GAP}",
        "ransac_refit": False,
        "match_details": (
            "All one-way Lowe-ratio-tested correspondences saved in gzip CSV"
        ),
        "reference_caveat": (
            "COLMAP-derived comparative reference, not absolute ground truth. "
            "AQUALOC's original reconstruction used exhaustive SIFT matching; "
            "possible reference-method advantage must be disclosed."
        ),
        "original_60774_row_benchmark_modified": False,
        "outputs": {
            "metrics": str(metrics_path),
            "summary": str(summary_path)
        },
        "extraction_summary": extraction_summary
    }

    metadata_path = OUT / "sequence6_reference_feature_metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, indent=4),
        encoding="utf-8"
    )

    print("\n" + "=" * 76)
    print("BENCHMARK COMPLETE")
    print("=" * 76)

    print("Metric rows:", len(all_metrics))
    print("Expected rows:", len(pairs) * 3)

    print("\nOUTPUTS")
    print("-" * 76)
    print(metrics_path)
    print(summary_path)
    print(metadata_path)

    print("\nCOMPRESSED MATCH COORDINATES")
    print("-" * 76)

    for item in extraction_summary:
        print(
            item["correspondence_file"],
            "|",
            item["correspondence_file_size_bytes"],
            "bytes"
        )

    print("\nOriginal benchmark modified: NO")
    print("Calibrated reference manifest modified: NO")
    print("=" * 76)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"\n[ERROR] {exc}", file=sys.stderr)
        sys.exit(1)
