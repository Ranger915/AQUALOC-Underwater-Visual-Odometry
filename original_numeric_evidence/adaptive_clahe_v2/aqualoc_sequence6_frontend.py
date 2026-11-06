import argparse
import csv
import json
import re
import time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

import adaptive_clahe as ac


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


def adaptive_decisions(paths, output, decide):
    """One pass over the images: per-frame decisions, cached for every detector."""
    decisions = {}
    for path in paths:
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            continue
        start = time.perf_counter()
        d = decide(image)
        d["stats_time_s"] = round(time.perf_counter() - start, 6)
        d["frame_id"] = frame_id(path)
        d["frame"] = path.name
        decisions[path.name] = d

    fields = ["frame_id", "frame", "global_std", "mean_local_std", "mode",
              "clip_limit", "tile_grid", "stats_time_s"]
    with (output / "frame_decisions.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(decisions.values())

    n = len(decisions)
    counts = Counter(d["mode"] for d in decisions.values())
    dist = {m: {"count": counts[m], "percent": round(100 * counts[m] / n, 2) if n else 0.0}
            for m in (ac.RAW, ac.MODERATE, ac.STRONG)}
    (output / "decision_distribution.json").write_text(
        json.dumps({"frames": n, "distribution": dist}, indent=2), encoding="utf-8")
    print(f"\nAdaptive decisions over {n} frames:")
    for m, v in dist.items():
        print(f"  {m}: {v['count']} ({v['percent']}%)")
    return decisions, dist


def write_summary(rows, output):
    groups = {}
    for r in rows:
        groups.setdefault((r["detector"], r["pair_stride"]), []).append(r)
    cols = ["keypoints_a", "keypoints_b", "ratio_matches", "ransac_inliers", "inlier_ratio",
            "extraction_time_pair_s", "matching_time_s", "geometry_time_s"]
    out = []
    for (det, stride), g in sorted(groups.items()):
        ok = [r for r in g if r["status"] == "ok"]
        row = {"detector": det, "pair_stride": stride, "pairs": len(g), "ok_pairs": len(ok)}
        for c in cols:
            row["mean_" + c] = round(float(np.mean([r[c] for r in g])), 6)
        e = "mean_symmetric_epipolar_error_px"
        row[e] = round(float(np.mean([r[e] for r in ok])), 6) if ok else ""
        out.append(row)
    if out:
        with (output / "adaptive_summary.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=out[0].keys())
            writer.writeheader()
            writer.writerows(out)


def integrity_report(rows, paths, strides, decisions, baseline_csv, output):
    checks = {}
    readable = {i for i, p in enumerate(paths) if p.name in decisions}
    expected = sum(1 for i in readable for s in strides if i - s in readable)
    for det in sorted({r["detector"] for r in rows}):
        n = sum(r["detector"] == det for r in rows)
        checks[f"pair_count_{det}"] = {"pass": n == expected, "rows": n, "expected": expected}
    checks["modes_sum_to_frames"] = {
        "pass": all(d["mode"] in (ac.RAW, ac.MODERATE, ac.STRONG) for d in decisions.values())}
    checks["output_dir_has_no_baseline_csv"] = {
        "pass": not (output / "feature_pair_metrics.csv").exists()}

    if baseline_csv:
        with open(baseline_csv, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            base = list(reader)
            header = reader.fieldnames
        key = lambda r: (r["detector"], r["frame_a"], r["frame_b"])
        base_raw = {key(r): r for r in base if r["preprocessing"] == "raw"}
        base_clahe = {key(r): r for r in base if r["preprocessing"] == "clahe"}
        ours = {key(r): r for r in rows}
        checks["columns_match_baseline"] = {"pass": header == list(rows[0].keys())}
        same = set(ours) == set(base_raw) == set(base_clahe)
        checks["same_pairs_as_baseline"] = {
            "pass": same, "adaptive": len(ours), "raw": len(base_raw), "clahe": len(base_clahe)}
        if same:
            # Both frames RAW (or both moderate) means identical input images, so
            # detection and matching counts must reproduce the baseline exactly.
            cols = ("keypoints_a", "keypoints_b", "raw_knn_matches", "ratio_matches")
            compared = mismatched = 0
            for k, r in ours.items():
                ma, mb = decisions[r["frame_a"]]["mode"], decisions[r["frame_b"]]["mode"]
                ref = base_raw if ma == mb == ac.RAW else base_clahe if ma == mb == ac.MODERATE else None
                if ref is None:
                    continue
                compared += 1
                mismatched += any(str(r[c]) != ref[k][c] for c in cols)
            checks["identical_input_pairs_reproduce_baseline"] = {
                "pass": mismatched == 0, "compared": compared, "mismatched": mismatched}
    report = {"all_passed": all(c["pass"] for c in checks.values()), "checks": checks}
    (output / "integrity_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Integrity: {'PASS' if report['all_passed'] else 'FAIL'} (see integrity_report.json)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-dir", required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--max-features", type=int, default=1500)
    parser.add_argument("--pair-strides", nargs="+", type=int, default=[1, 5, 20])
    parser.add_argument("--adaptive", action="store_true",
                        help="run only adaptive_clahe, into a separate results folder")
    parser.add_argument("--adaptive-version", type=int, choices=(1, 2), default=1,
                        help="1 = global+local gate (frozen), 2 = local-only gate")
    parser.add_argument("--decisions-only", action="store_true",
                        help="with --adaptive: write per-frame decisions and their distribution, skip matching")
    parser.add_argument("--baseline-csv", default=None,
                        help="baseline feature_pair_metrics.csv for the integrity report")
    args = parser.parse_args()
    if args.decisions_only and not args.adaptive:
        parser.error("--decisions-only requires --adaptive")

    image_dir = Path(args.image_dir)
    if not image_dir.is_dir():
        raise FileNotFoundError(f"Image folder not found: {image_dir}")

    output = Path(args.output_dir) if args.output_dir else (
        image_dir.parent / (
            ("sequence6_adaptive_clahe_results" if args.adaptive_version == 1
             else "sequence6_adaptive_clahe_v2_results") if args.adaptive
            else "sequence6_frontend_results"
        )
    )
    output.mkdir(parents=True, exist_ok=True)
    if args.adaptive and (output / "feature_pair_metrics.csv").exists():
        raise RuntimeError(f"{output} holds baseline results; use a separate --output-dir")

    paths = sorted(
        [p for p in image_dir.glob("*.png") if frame_id(p) >= 0],
        key=frame_id
    )
    if args.max_frames:
        paths = paths[:args.max_frames]
    if not paths:
        raise RuntimeError("No frame<number>.png images found.")

    label = "adaptive_clahe" if args.adaptive_version == 1 else "adaptive_clahe_v2"
    modes = (label,) if args.adaptive else ("raw", "clahe")
    decisions, distribution = {}, None
    if args.adaptive:
        decisions, distribution = adaptive_decisions(
            paths, output, ac.decide if args.adaptive_version == 1 else ac.decide_v2)
        if args.decisions_only:
            return

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
    ransac_exceptions = 0

    print(f"Found {len(paths)} images")
    print(f"Detectors: {list(detectors)}")
    print(f"Output: {output}")

    for name, detector in detectors.items():
        norm = cv2.NORM_HAMMING if name != "SIFT" else cv2.NORM_L2
        matcher = cv2.BFMatcher(norm)
        for preprocessing in modes:
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
                elif preprocessing == label:
                    image = ac.enhance(image, decisions[path.name]["clip_limit"])

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
                            try:
                                F, mask = cv2.findFundamentalMat(
                                    p1, p2, cv2.FM_RANSAC, 1.5, 0.99
                                )
                            except cv2.error:
                                # OpenCV 4.13 asserts when p1 == p2 (zero-motion matches).
                                # Recorded as fundamental_matrix_failed and counted in metadata.
                                F, mask = None, None
                                ransac_exceptions += 1
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

    csv_path = output / (
        "adaptive_pair_metrics.csv" if args.adaptive else "feature_pair_metrics.csv"
    )
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
        "preprocessing": list(modes),
        "pair_strides": strides,
        "max_features": args.max_features,
        "opencv_version": cv2.__version__,
        "numpy_version": np.__version__,
        "metrics_rows": len(rows),
        "ransac_exceptions": ransac_exceptions,
        "note": "Uncalibrated feature benchmark, not a complete VO/SLAM system. Epipolar error is a point-to-line residual proxy."
    }
    if args.adaptive:
        cfg = ac.config() if args.adaptive_version == 1 else ac.config_v2()
        metadata[label] = {**cfg, "decision_distribution": distribution}
    (output / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )

    print("\nBenchmark completed.")
    print(f"CSV: {csv_path}")
    print(f"Metadata: {output / 'run_metadata.json'}")
    print(f"Total rows: {len(rows)}")

    if args.adaptive and rows:
        write_summary(rows, output)
        integrity_report(rows, paths, strides, decisions, args.baseline_csv, output)


if __name__ == "__main__":
    main()
