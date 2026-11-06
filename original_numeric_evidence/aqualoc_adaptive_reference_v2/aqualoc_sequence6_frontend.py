import argparse
import csv
import json
import re
import time
from pathlib import Path

import cv2
import numpy as np


def frame_id(path):
    m = re.search(r"frame(\d+)", path.stem, re.I)
    return int(m.group(1)) if m else -1


def detect(detector, image, limit):
    start = time.perf_counter()
    kp, desc = detector.detectAndCompute(image, None)
    elapsed = time.perf_counter() - start

    if kp and len(kp) > limit:
        idx = np.argsort([k.response for k in kp])[::-1][:limit]
        kp = [kp[i] for i in idx]
        desc = desc[idx]

    return kp or [], desc, elapsed


def grid_coverage(points, width, height):
    cells = set()
    for x, y in points:
        cx = min(3, int(x / width * 4))
        cy = min(3, int(y / height * 4))
        cells.add((cx, cy))
    return len(cells)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-dir", required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--max-features", type=int, default=1500)
    parser.add_argument("--pair-strides", nargs="+", type=int, default=[1, 5, 20])
    args = parser.parse_args()

    image_dir = Path(args.image_dir)
    if not image_dir.is_dir():
        raise FileNotFoundError(f"Image folder not found: {image_dir}")

    output = Path(args.output_dir) if args.output_dir else (
        image_dir.parent / "sequence6_frontend_results"
    )
    output.mkdir(parents=True, exist_ok=True)

    paths = sorted(
        [p for p in image_dir.glob("*.png") if frame_id(p) >= 0],
        key=frame_id
    )
    if args.max_frames:
        paths = paths[:args.max_frames]
    if not paths:
        raise RuntimeError("No frame<number>.png images found.")

    # OpenCV 4 commonly exposes AKAZE_create in cv2. Some OpenCV 5
    # builds expose it through cv2.xfeatures2d instead.
    if hasattr(cv2, "AKAZE_create"):
        akaze_factory = cv2.AKAZE_create
    elif hasattr(cv2, "xfeatures2d") and hasattr(cv2.xfeatures2d, "AKAZE_create"):
        akaze_factory = cv2.xfeatures2d.AKAZE_create
    else:
        raise RuntimeError(
            "AKAZE is unavailable in this OpenCV installation. "
            "Install opencv-contrib-python or use an OpenCV build with AKAZE."
        )

    akaze_detector = akaze_factory(
        descriptor_type=cv2.AKAZE_DESCRIPTOR_MLDB
    )

    detectors = {
        "ORB": cv2.ORB_create(nfeatures=args.max_features),
        "AKAZE_MLDB": akaze_detector
    }
    if hasattr(cv2, "SIFT_create"):
        detectors["SIFT"] = cv2.SIFT_create(nfeatures=args.max_features)

    strides = sorted(set(args.pair_strides))
    rows = []

    print(f"Found {len(paths)} images")
    print(f"Detectors: {list(detectors)}")
    print(f"Output: {output}")

    for name, detector in detectors.items():
        norm = cv2.NORM_HAMMING if name != "SIFT" else cv2.NORM_L2
        matcher = cv2.BFMatcher(norm)
        for preprocessing in ("raw", "clahe"):
            cache = {}
            print(f"\nRunning {name} / {preprocessing}")

            for i, path in enumerate(paths):
                image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
                if image is None:
                    print(f"Skipping unreadable image: {path.name}")
                    continue

                if preprocessing == "clahe":
                    image = cv2.createCLAHE(
                        clipLimit=2.0, tileGridSize=(8, 8)
                    ).apply(image)

                kp, desc, extract_time = detect(
                    detector, image, args.max_features
                )
                cache[i] = (kp, desc, image.shape, extract_time)

                for stride in strides:
                    j = i - stride
                    if j not in cache:
                        continue

                    kp1, d1, shape, t1 = cache[j]
                    kp2, d2, _, t2 = cache[i]

                    row = {
                        "detector": name,
                        "preprocessing": preprocessing,
                        "frame_a": paths[j].name,
                        "frame_b": path.name,
                        "frame_a_id": frame_id(paths[j]),
                        "frame_b_id": frame_id(path),
                        "pair_stride": stride,
                        "keypoints_a": len(kp1),
                        "keypoints_b": len(kp2),
                        "raw_knn_matches": 0,
                        "ratio_matches": 0,
                        "ransac_inliers": 0,
                        "inlier_ratio": 0.0,
                        "mean_symmetric_epipolar_error_px": "",
                        "grid_cells_covered_4x4": 0,
                        "extraction_time_pair_s": round(t1 + t2, 6),
                        "matching_time_s": 0.0,
                        "geometry_time_s": 0.0,
                        "status": "insufficient_descriptors"
                    }

                    if d1 is not None and d2 is not None and len(d1) >= 2 and len(d2) >= 2:
                        start = time.perf_counter()
                        knn = matcher.knnMatch(d1, d2, k=2)
                        pairs = [p for p in knn if len(p) == 2]
                        good = [m for m, n in pairs if m.distance < 0.75 * n.distance]
                        row["raw_knn_matches"] = len(pairs)
                        row["ratio_matches"] = len(good)
                        row["matching_time_s"] = round(time.perf_counter() - start, 6)

                        if len(good) >= 8:
                            p1 = np.float32([kp1[m.queryIdx].pt for m in good])
                            p2 = np.float32([kp2[m.trainIdx].pt for m in good])

                            start = time.perf_counter()
                            F, mask = cv2.findFundamentalMat(
                                p1, p2, cv2.FM_RANSAC, 1.5, 0.99
                            )
                            row["geometry_time_s"] = round(time.perf_counter() - start, 6)

                            if F is not None and mask is not None and F.shape == (3, 3):
                                keep = mask.ravel().astype(bool)
                                a, b = p1[keep], p2[keep]
                                count = int(keep.sum())
                                row["ransac_inliers"] = count
                                row["inlier_ratio"] = round(count / len(good), 6)

                                if count >= 8:
                                    x1 = np.column_stack((a, np.ones(count)))
                                    x2 = np.column_stack((b, np.ones(count)))
                                    l2 = (F @ x1.T).T
                                    l1 = (F.T @ x2.T).T
                                    num = np.abs(np.sum(x2 * l2, axis=1))
                                    d_a = num / np.maximum(
                                        np.sqrt(l1[:, 0]**2 + l1[:, 1]**2), 1e-12
                                    )
                                    d_b = num / np.maximum(
                                        np.sqrt(l2[:, 0]**2 + l2[:, 1]**2), 1e-12
                                    )
                                    row["mean_symmetric_epipolar_error_px"] = round(
                                        float(np.mean((d_a + d_b) / 2)), 6
                                    )
                                    row["grid_cells_covered_4x4"] = grid_coverage(
                                        a, shape[1], shape[0]
                                    )
                                    row["status"] = "ok"
                                else:
                                    row["status"] = "few_ransac_inliers"
                            else:
                                row["status"] = "fundamental_matrix_failed"
                        else:
                            row["status"] = "fewer_than_8_ratio_matches"

                    rows.append(row)

                oldest = i - max(strides)
                cache = {k: v for k, v in cache.items() if k >= oldest}

                if (i + 1) % 100 == 0 or i + 1 == len(paths):
                    print(f"  {i + 1}/{len(paths)} frames")

    csv_path = output / "feature_pair_metrics.csv"
    if rows:
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)

    metadata = {
        "dataset": "AQUALOC Sequence 06",
        "image_count": len(paths),
        "first_image": paths[0].name,
        "last_image": paths[-1].name,
        "detectors": list(detectors),
        "preprocessing": ["raw", "clahe"],
        "pair_strides": strides,
        "max_features": args.max_features,
        "opencv_version": cv2.__version__,
        "numpy_version": np.__version__,
        "metrics_rows": len(rows),
        "note": "Uncalibrated feature benchmark, not a complete VO/SLAM system. Epipolar error is a point-to-line residual proxy."
    }
    (output / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )

    print("\nBenchmark completed.")
    print(f"CSV: {csv_path}")
    print(f"Metadata: {output / 'run_metadata.json'}")
    print(f"Total rows: {len(rows)}")


if __name__ == "__main__":
    main()
