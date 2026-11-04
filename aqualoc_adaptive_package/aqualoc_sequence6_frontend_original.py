#!/usr/bin/env python3
"""
AQUALOC Sequence 06 feature-front-end benchmark.
Reads frame*.png images without modifying them.
Compares ORB, AKAZE-MLDB and SIFT on raw grayscale and CLAHE images.
Writes pair-level CSV metrics and run metadata to a separate output folder.
"""
import argparse
import csv
import json
import re
import sys
import time
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np


def frame_number(path: Path):
    m = re.search(r"frame(\d+)", path.stem, re.IGNORECASE)
    return int(m.group(1)) if m else None


def make_detector(name, max_features):
    if name == "ORB":
        return cv2.ORB_create(nfeatures=max_features), cv2.NORM_HAMMING
    if name == "AKAZE":
        return cv2.AKAZE_create(), cv2.NORM_HAMMING
    if name == "SIFT":
        if not hasattr(cv2, "SIFT_create"):
            raise RuntimeError("SIFT is unavailable in this OpenCV build.")
        return cv2.SIFT_create(nfeatures=max_features), cv2.NORM_L2
    raise ValueError(name)


def prepare_image(gray, preprocessing):
    if preprocessing == "raw":
        return gray
    if preprocessing == "clahe":
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        return clahe.apply(gray)
    raise ValueError(preprocessing)


def extract(detector, gray, max_features):
    t0 = time.perf_counter()
    keypoints, descriptors = detector.detectAndCompute(gray, None)
    elapsed = time.perf_counter() - t0
    if descriptors is not None and len(keypoints) > max_features:
        order = np.argsort([-kp.response for kp in keypoints])[:max_features]
        keypoints = [keypoints[i] for i in order]
        descriptors = descriptors[order]
    return keypoints, descriptors, elapsed


def grid_coverage(points, width, height, grid=4):
    if len(points) == 0:
        return 0.0
    xs = np.clip((points[:, 0] / max(width, 1) * grid).astype(int), 0, grid - 1)
    ys = np.clip((points[:, 1] / max(height, 1) * grid).astype(int), 0, grid - 1)
    return len(set(zip(xs.tolist(), ys.tolist()))) / float(grid * grid)


def compare_pair(kp1, des1, kp2, des2, norm, shape):
    h, w = shape[:2]
    result = {
        "keypoints_1": len(kp1), "keypoints_2": len(kp2),
        "raw_knn_matches": 0, "ratio_matches": 0, "ransac_inliers": 0,
        "inlier_ratio": 0.0, "mean_symmetric_epipolar_error_px": "",
        "grid_coverage_1": 0.0, "grid_coverage_2": 0.0,
        "matching_time_s": 0.0, "geometry_time_s": 0.0,
        "status": "ok"
    }
    if des1 is None or des2 is None or len(des1) < 2 or len(des2) < 2:
        result["status"] = "insufficient_descriptors"
        return result

    matcher = cv2.BFMatcher(norm)
    t0 = time.perf_counter()
    knn = matcher.knnMatch(des1, des2, k=2)
    result["matching_time_s"] = time.perf_counter() - t0
    good = [m for pair in knn if len(pair) == 2
            for m, n in [pair] if m.distance < 0.75 * n.distance]
    result["raw_knn_matches"] = len(knn)
    result["ratio_matches"] = len(good)
    if len(good) < 8:
        result["status"] = "fewer_than_8_ratio_matches"
        return result

    pts1 = np.float32([kp1[m.queryIdx].pt for m in good])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in good])
    t0 = time.perf_counter()
    F, mask = cv2.findFundamentalMat(
        pts1, pts2, cv2.FM_RANSAC, 1.5, 0.99
    )
    result["geometry_time_s"] = time.perf_counter() - t0
    if F is None or mask is None:
        result["status"] = "fundamental_matrix_failed"
        return result

    inliers = mask.ravel().astype(bool)
    count = int(inliers.sum())
    result["ransac_inliers"] = count
    result["inlier_ratio"] = count / max(len(good), 1)
    if count:
        p1, p2 = pts1[inliers], pts2[inliers]
        result["grid_coverage_1"] = grid_coverage(p1, w, h)
        result["grid_coverage_2"] = grid_coverage(p2, w, h)
        # Symmetric point-to-epipolar-line distance in pixels.
        x1 = np.column_stack([p1, np.ones(len(p1))])
        x2 = np.column_stack([p2, np.ones(len(p2))])
        lines2 = (F @ x1.T).T
        lines1 = (F.T @ x2.T).T
        numer = np.abs(np.sum(x2 * lines2, axis=1))
        d2 = numer / np.maximum(np.linalg.norm(lines2[:, :2], axis=1), 1e-12)
        d1 = numer / np.maximum(np.linalg.norm(lines1[:, :2], axis=1), 1e-12)
        result["mean_symmetric_epipolar_error_px"] = float(np.mean((d1 + d2) * 0.5))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-dir", required=True, help="Folder containing frame000000.png ...")
    parser.add_argument("--output-dir", default="", help="Separate output folder; default is a sibling results folder")
    parser.add_argument("--max-features", type=int, default=1500)
    parser.add_argument("--pair-strides", type=int, nargs="+", default=[1, 5, 20],
                        help="Temporal image offsets to compare; default: 1 5 20")
    parser.add_argument("--max-frames", type=int, default=0,
                        help="Optional smoke-test limit; 0 processes all frames")
    args = parser.parse_args()

    image_dir = Path(args.image_dir).expanduser().resolve()
    if not image_dir.is_dir():
        raise FileNotFoundError(f"Image folder not found: {image_dir}")
    output_dir = (Path(args.output_dir).expanduser().resolve() if args.output_dir
                  else image_dir.parent / "sequence6_frontend_results")
    output_dir.mkdir(parents=True, exist_ok=True)

    files = []
    for f in image_dir.glob("*.png"):
        n = frame_number(f)
        if n is not None:
            files.append((n, f))
    files.sort(key=lambda x: x[0])
    if args.max_frames > 0:
        files = files[:args.max_frames]
    if len(files) < 2:
        raise RuntimeError(f"Found only {len(files)} matching PNG files in {image_dir}")
    ids = [n for n, _ in files]
    if len(set(ids)) != len(ids):
        raise RuntimeError("Duplicate frame numbers detected.")
    strides = sorted(set(args.pair_strides))
    if any(s < 1 for s in strides):
        raise ValueError("All pair strides must be >= 1")

    methods = [("raw", "ORB"), ("raw", "AKAZE"), ("raw", "SIFT"),
               ("clahe", "ORB"), ("clahe", "AKAZE"), ("clahe", "SIFT")]
    csv_path = output_dir / "feature_pair_metrics.csv"
    metadata_path = output_dir / "run_metadata.json"
    rows_written = 0
    skipped_sift = False

    metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "image_dir": str(image_dir), "output_dir": str(output_dir),
        "image_count": len(files), "first_frame_id": ids[0], "last_frame_id": ids[-1],
        "pair_strides": strides, "max_features": args.max_features,
        "preprocessing": ["raw", "CLAHE (clipLimit=2.0, tileGridSize=8x8)"],
        "geometry": "Fundamental matrix RANSAC; reprojection proxy is mean symmetric epipolar distance",
        "ratio_test": 0.75, "ransac_threshold_px": 1.5,
        "note": "This is a feature-correspondence benchmark, not a complete VO/SLAM trajectory."
    }

    columns = [
        "preprocessing", "detector", "frame_id_1", "frame_id_2", "frame_stride",
        "keypoints_1", "keypoints_2", "raw_knn_matches", "ratio_matches",
        "ransac_inliers", "inlier_ratio", "mean_symmetric_epipolar_error_px",
        "grid_coverage_1", "grid_coverage_2", "extract_time_1_s", "extract_time_2_s",
        "matching_time_s", "geometry_time_s", "status"
    ]

    with csv_path.open("w", newline="", encoding="utf-8") as fcsv:
        writer = csv.DictWriter(fcsv, fieldnames=columns)
        writer.writeheader()

        for preprocessing, detector_name in methods:
            try:
                detector, norm = make_detector(detector_name, args.max_features)
            except RuntimeError as exc:
                if detector_name == "SIFT":
                    skipped_sift = True
                    print(f"[WARN] {exc} Skipping SIFT.", flush=True)
                    continue
                raise

            print(f"\n[RUN] {preprocessing.upper()} + {detector_name}", flush=True)
            cache = OrderedDict()
            for idx, (frame_id, path) in enumerate(files):
                gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
                if gray is None:
                    print(f"[WARN] Could not read {path}; skipping frame.", flush=True)
                    continue
                processed = prepare_image(gray, preprocessing)
                kp, des, extract_time = extract(detector, processed, args.max_features)
                cache[frame_id] = (kp, des, extract_time, processed.shape)
                min_keep = frame_id - max(strides)
                for old_id in list(cache.keys()):
                    if old_id < min_keep:
                        del cache[old_id]

                for stride in strides:
                    prev_id = frame_id - stride
                    if prev_id not in cache:
                        continue
                    kp1, des1, t1, shape1 = cache[prev_id]
                    metrics = compare_pair(kp1, des1, kp, des, norm, shape1)
                    row = {
                        "preprocessing": preprocessing, "detector": detector_name,
                        "frame_id_1": prev_id, "frame_id_2": frame_id, "frame_stride": stride,
                        **metrics,
                        "extract_time_1_s": f"{t1:.6f}",
                        "extract_time_2_s": f"{extract_time:.6f}"
                    }
                    writer.writerow(row)
                    rows_written += 1
                if (idx + 1) % 250 == 0 or idx + 1 == len(files):
                    fcsv.flush()
                    print(f"  processed {idx + 1}/{len(files)} frames", flush=True)

    metadata["rows_written"] = rows_written
    metadata["sift_skipped"] = skipped_sift
    metadata["csv_file"] = str(csv_path)
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print("\n[DONE] Feature benchmark completed.")
    print(f"[OUTPUT] {csv_path}")
    print(f"[OUTPUT] {metadata_path}")
    print("[NOTE] Share these two small files for analysis; do not upload the image folder.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        sys.exit(1)
