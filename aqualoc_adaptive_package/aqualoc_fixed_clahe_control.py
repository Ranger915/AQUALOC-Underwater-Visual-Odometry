#!/usr/bin/env python3
"""Non-destructive adaptive-only frontend benchmark, reusing original feature metrics."""
import argparse
import csv
import json
import hashlib
import time
from collections import Counter, OrderedDict
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
import cv2
from adaptive_clahe import AdaptiveConfig, apply_adaptive, decide
from aqualoc_sequence6_frontend_original import frame_number, make_detector, extract, compare_pair

COLUMNS = ["preprocessing", "detector", "frame_id_1", "frame_id_2", "frame_stride",
           "keypoints_1", "keypoints_2", "raw_knn_matches", "ratio_matches",
           "ransac_inliers", "inlier_ratio", "mean_symmetric_epipolar_error_px",
           "grid_coverage_1", "grid_coverage_2", "extract_time_1_s", "extract_time_2_s",
           "matching_time_s", "geometry_time_s", "status"]
DECISION_COLUMNS = ["frame_id", "filename", "global_std", "local_std", "decision", "clip_limit", "version"]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--image-dir", required=True)
    p.add_argument("--output-dir", default=None)
    p.add_argument("--version", choices=["v1", "v2"], default="v2")
    p.add_argument("--max-features", type=int, default=1500)
    p.add_argument("--pair-strides", nargs="+", type=int, default=[1, 5, 20])
    p.add_argument("--max-frames", type=int, default=0)
    p.add_argument("--v2-local-threshold", type=float, default=10.5)
    args = p.parse_args()
    image_dir = Path(args.image_dir).expanduser().resolve()
    if not image_dir.is_dir():
        raise FileNotFoundError(image_dir)
    output = (Path(args.output_dir).expanduser().resolve() if args.output_dir else
              image_dir.parent / f"sequence6_adaptive_{args.version}_results")
    if output == image_dir or output.name == "sequence6_frontend_results":
        raise ValueError("Output directory must not be original image or baseline results folder")
    output.mkdir(parents=True, exist_ok=True)
    for name in ("adaptive_feature_pair_metrics.csv", "adaptive_frame_decisions.csv", "adaptive_metadata.json"):
        if (output / name).exists():
            raise FileExistsError(f"Refusing to overwrite existing experiment output: {output / name}")
    files = sorted(((frame_number(f), f) for f in image_dir.glob("*.png") if frame_number(f) is not None), key=lambda x:x[0])
    if args.max_frames:
        files = files[:args.max_frames]
    if len(files) < 2 or len(set(n for n, _ in files)) != len(files):
        raise ValueError("Need >=2 images and unique frame IDs")
    strides = sorted(set(args.pair_strides))
    if any(s < 1 for s in strides):
        raise ValueError("Strides must be positive")
    config = AdaptiveConfig(version=args.version, local_threshold_v2=args.v2_local_threshold)
    decisions = {}
    decision_counts = Counter()
    decision_path = output / "adaptive_frame_decisions.csv"
    with decision_path.open("x", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=DECISION_COLUMNS)
        writer.writeheader()
        for i, (fid, path) in enumerate(files):
            gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if gray is None:
                raise RuntimeError(f"Unreadable frame: {path}")
            d = {"global_std": "", "local_std": "", "decision": "fixed_clahe", "clip_limit": 2.0, "version": "control"}
            decisions[fid] = d
            decision_counts[d["decision"]] += 1
            writer.writerow({"frame_id": fid, "filename": path.name, **d})
            if (i + 1) % 500 == 0:
                print(f"[DECISIONS] {i+1}/{len(files)}", flush=True)
    print("[DECISION DISTRIBUTION]", dict(decision_counts), flush=True)
    counts = Counter()
    metrics_path = output / "adaptive_feature_pair_metrics.csv"
    with metrics_path.open("x", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        for detector_name in ("ORB", "AKAZE", "SIFT"):
            try:
                detector, norm = make_detector(detector_name, args.max_features)
            except (RuntimeError, AttributeError) as e:
                if detector_name == "AKAZE" and hasattr(cv2, "xfeatures2d") and hasattr(cv2.xfeatures2d, "AKAZE_create"):
                    detector, norm = cv2.xfeatures2d.AKAZE_create(), cv2.NORM_HAMMING
                else:
                    raise RuntimeError(f"Missing detector {detector_name}: {e}") from e
            cache = OrderedDict()
            for i, (fid, path) in enumerate(files):
                gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
                if gray is None:
                    raise RuntimeError(f"Unreadable frame: {path}")
                processed = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
                kp, des, elapsed = extract(detector, processed, args.max_features)
                cache[fid] = (kp, des, elapsed, processed.shape)
                for old_id in list(cache):
                    if old_id < fid - max(strides):
                        del cache[old_id]
                for stride in strides:
                    prev = fid - stride
                    if prev not in cache:
                        continue
                    kp1, des1, t1, shape = cache[prev]
                    m = compare_pair(kp1, des1, kp, des, norm, shape)
                    writer.writerow({"preprocessing": "fixed_clahe_control", "detector": detector_name,
                                     "frame_id_1": prev, "frame_id_2": fid, "frame_stride": stride,
                                     **m, "extract_time_1_s": f"{t1:.6f}", "extract_time_2_s": f"{elapsed:.6f}"})
                    counts[detector_name] += 1
                if (i+1) % 250 == 0 or i+1 == len(files):
                    f.flush()
                    print(f"[{detector_name}] {i+1}/{len(files)} frames", flush=True)
    expected = sum(sum((fid-stride) in decisions for fid in decisions) for stride in strides)
    if any(counts[d] != expected for d in ("ORB", "AKAZE", "SIFT")):
        raise AssertionError(f"Pair count mismatch: {dict(counts)}, expected {expected}")
    source_path = Path(__file__).with_name("aqualoc_sequence6_frontend_original.py")
    metadata = {"created_utc": datetime.now(timezone.utc).isoformat(), "dataset": "AQUALOC Sequence 06",
                "experiment": "fixed_clahe_control", "image_dir": str(image_dir),
                "image_count": len(files), "frame_ids": [files[0][0], files[-1][0]],
                "decision_counts": dict(decision_counts), "detector_pair_counts": dict(counts),
                "expected_pairs_per_detector": expected, "parameters": asdict(config),
                "max_features": args.max_features, "pair_strides": strides,
                "ratio_test": 0.75, "ransac_threshold_px": 1.5, "ransac_confidence": 0.99,
                "fixed_clahe_reference": {"clipLimit": 2.0, "tileGridSize": [8,8]},
                "original_frontend_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
                "opencv_version": cv2.__version__,
                "limitation": "Uncalibrated frontend only; no independent precision, recall, F1, AP, Sampson-reference scoring or trajectory ATE.",
                "v2_rule_warning": "v2 local-std-only rule inferred from notes; verify against original v2 implementation before publication."}
    (output / "adaptive_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print("[DONE]", metrics_path, decision_path, flush=True)

if __name__ == "__main__":
    main()
