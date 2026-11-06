#!/usr/bin/env python3
"""Paired keyframe-anchor moving-block bootstrap for AQUALOC mutual-match AP.

Uses the same reference pair IDs, overlap labels, and mutual 2px scores for
fixed CLAHE and adaptive_v2. Implements non-interpolated binary AP with ties,
equivalent to sklearn average_precision_score for samples with both classes.
No third-party dependencies required.
"""
import argparse
import csv
import json
import math
import random
import statistics
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

DETECTORS = ("ORB", "AKAZE_MLDB", "SIFT")
SCOPES = ("all_pairs", "local_gap_le_100", "nonlocal_gap_ge_200")
METRICS = (("average_precision_count", "mutual_inliers_2px"),
           ("average_precision_fraction", "mutual_fraction_2px"))


def read_rows(path, method):
    out = {}
    required = {"detector", "preprocessing", "frame_id_i", "frame_id_j",
                "overlap_label", "mutual_inliers_2px", "mutual_fraction_2px"}
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Missing fields in {path}: {sorted(missing)}")
        for row in reader:
            if row["preprocessing"].strip().lower() != method:
                continue
            det = row["detector"].strip().upper()
            if det not in DETECTORS:
                raise ValueError(f"Unknown detector {det!r}")
            a, b = int(row["frame_id_i"]), int(row["frame_id_j"])
            if not a < b:
                raise ValueError(f"Invalid or reversed pair ({a},{b})")
            key = (det, a, b)
            if key in out:
                raise ValueError(f"Duplicate row {key}")
            label = int(row["overlap_label"])
            if label not in (0, 1):
                raise ValueError(f"Invalid label for {key}: {label}")
            count = float(row["mutual_inliers_2px"])
            frac = float(row["mutual_fraction_2px"])
            if not all(math.isfinite(x) for x in (count, frac)) or count < 0 or frac < 0 or frac > 1.000001:
                raise ValueError(f"Invalid score for {key}: {count}, {frac}")
            out[key] = (label, count, frac)
    return out


def read_summary(path, method):
    out = {}
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        expected = {"detector", "preprocessing", "scope", "sampson_threshold_px",
                    "minimum_mutual_inliers", "average_precision_count", "average_precision_fraction"}
        missing = expected - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Missing summary fields: {sorted(missing)}")
        for r in reader:
            if (r["preprocessing"].strip().lower() == method
                and float(r["sampson_threshold_px"]) == 2
                and int(r["minimum_mutual_inliers"]) == 8):
                key = (r["detector"].strip().upper(), r["scope"])
                for metric, _ in METRICS:
                    val = float(r[metric])
                    if key in out and metric in out[key] and abs(out[key][metric] - val) > 1e-12:
                        raise ValueError(f"Conflicting summary AP for {key}")
                    out.setdefault(key, {})[metric] = val
    return out


def scope_ok(key, scope):
    gap = key[2] - key[1]
    return scope == "all_pairs" or (scope == "local_gap_le_100" and gap <= 100) or (scope == "nonlocal_gap_ge_200" and gap >= 200)


def percentile(x, q):
    x = sorted(x)
    z = q * (len(x) - 1)
    lo = int(z)
    hi = min(lo + 1, len(x) - 1)
    return x[lo] + (x[hi] - x[lo]) * (z - lo)


def ranking(records, keys, score_pos, anchor_index):
    # score, positive label, originating anchor index. Sorting is constant
    # across replicates, which greatly reduces bootstrap computation time.
    return sorted(((records[k][score_pos], records[k][0], anchor_index[k[1]])
                   for k in keys), key=lambda x: x[0], reverse=True)


def weighted_ap(sorted_rank, counts):
    """Non-interpolated weighted AP, same tie behavior as sklearn AP."""
    total_positive = sum(counts[anchor] for _, label, anchor in sorted_rank if label)
    if total_positive <= 0:
        return None
    tp = fp = last_tp = 0
    ap_sum = 0.0
    n = len(sorted_rank)
    for i, (score, label, anchor) in enumerate(sorted_rank):
        w = counts[anchor]
        if label:
            tp += w
        else:
            fp += w
        # Evaluate precision at the end of a tie group.
        if i == n - 1 or sorted_rank[i+1][0] != score:
            if tp + fp:
                ap_sum += (tp - last_tp) * (tp / (tp + fp))
            last_tp = tp
    return ap_sum / total_positive


def bootstrap_anchor_counts(n, block, rng):
    counts = [0] * n
    for _ in range((n + block - 1) // block):
        start = rng.randrange(n)
        for d in range(block):
            counts[(start + d) % n] += 1
    return counts


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--reps", type=int, default=1000)
    p.add_argument("--blocks", type=int, nargs="+", default=[10])
    p.add_argument("--seed", type=int, default=20261104)
    p.add_argument("--tolerance", type=float, default=1e-8)
    args = p.parse_args()
    if args.reps < 100 or not args.blocks or min(args.blocks) < 1:
        p.error("At least 100 replicates and positive block sizes are required")
    root = args.root.expanduser().resolve()
    base = root / "raw_data/sequence6_mutual_matching_results"
    adap = root / "sequence6_adaptive_v2_mutual_matching_results"
    fixed = read_rows(base / "sequence6_mutual_feature_metrics.csv", "clahe")
    adaptive = read_rows(adap / "sequence6_mutual_feature_metrics.csv", "adaptive_v2")
    ref_fix = read_summary(base / "sequence6_mutual_feature_summary.csv", "clahe")
    ref_adap = read_summary(adap / "sequence6_mutual_feature_summary.csv", "adaptive_v2")
    if not fixed or set(fixed) != set(adaptive):
        raise RuntimeError(f"Pair alignment failed: fixed {len(fixed)}, adaptive {len(adaptive)}")
    if len(fixed) != 43095:
        raise RuntimeError(f"Expected 43095 matched detector/pair records, got {len(fixed)}")
    if any(fixed[k][0] != adaptive[k][0] for k in fixed):
        raise RuntimeError("Reference overlap labels disagree between experiments")
    out = root / "sequence6_paper_validation"
    out.mkdir(exist_ok=True)
    results = []
    reproduced = []
    for det in DETECTORS:
        det_keys = [k for k in fixed if k[0] == det]
        anchor_ids = sorted({k[1] for k in det_keys})
        anchor_index = {a: i for i, a in enumerate(anchor_ids)}
        full_counts = [1] * len(anchor_ids)
        for scope in SCOPES:
            keys = [k for k in det_keys if scope_ok(k, scope)]
            labels = Counter(fixed[k][0] for k in keys)
            if not labels[0] or not labels[1]:
                raise RuntimeError(f"No positive or negative samples in {det}/{scope}")
            ranks = {}
            for metric, field in METRICS:
                pos = 1 if field == "mutual_inliers_2px" else 2
                ranks["clahe", metric] = ranking(fixed, keys, pos, anchor_index)
                ranks["adaptive_v2", metric] = ranking(adaptive, keys, pos, anchor_index)
                a = weighted_ap(ranks["clahe", metric], full_counts)
                b = weighted_ap(ranks["adaptive_v2", metric], full_counts)
                expected_a = ref_fix.get((det, scope), {}).get(metric)
                expected_b = ref_adap.get((det, scope), {}).get(metric)
                if expected_a is None or expected_b is None:
                    raise RuntimeError(f"Summary AP not found for {det}/{scope}/{metric}")
                reproduced.extend([
                    dict(detector=det, scope=scope, metric=metric, method="clahe", calculated_ap=a, summary_ap=expected_a, absolute_error=abs(a-expected_a)),
                    dict(detector=det, scope=scope, metric=metric, method="adaptive_v2", calculated_ap=b, summary_ap=expected_b, absolute_error=abs(b-expected_b))
                ])
                if abs(a - expected_a) > args.tolerance or abs(b - expected_b) > args.tolerance:
                    raise RuntimeError(f"AP DEFINITION MISMATCH for {det}/{scope}/{metric}: "
                                       f"recalculated {a:.12g}/{b:.12g} vs "
                                       f"summary {expected_a:.12g}/{expected_b:.12g}. "
                                       "No bootstrap was reported; inspect original scorer.")
            print(f"[VALIDATED AP] {det} / {scope} / {len(keys)} pairs", flush=True)
            for block in args.blocks:
                rng = random.Random(args.seed + block * 101 + DETECTORS.index(det) * 10007 + SCOPES.index(scope) * 1000003)
                deltas = {m: [] for m, _ in METRICS}
                for _ in range(args.reps):
                    counts = bootstrap_anchor_counts(len(anchor_ids), block, rng)
                    for metric, _ in METRICS:
                        a = weighted_ap(ranks["clahe", metric], counts)
                        b = weighted_ap(ranks["adaptive_v2", metric], counts)
                        if a is not None and b is not None:
                            deltas[metric].append(b - a)
                for metric, _ in METRICS:
                    a = weighted_ap(ranks["clahe", metric], full_counts)
                    b = weighted_ap(ranks["adaptive_v2", metric], full_counts)
                    diffs = deltas[metric]
                    if len(diffs) < args.reps * 0.9:
                        raise RuntimeError(f"Too few valid bootstrap replicates for {det}/{scope}/{metric}")
                    entry = {
                        "detector": det, "scope": scope, "metric": metric,
                        "fixed_ap": a, "adaptive_v2_ap": b,
                        "paired_difference_adaptive_minus_fixed": b - a,
                        "ci95_lower": percentile(diffs, .025),
                        "ci95_upper": percentile(diffs, .975),
                        "probability_difference_positive": sum(v > 0 for v in diffs) / len(diffs),
                        "bootstrap_reps_valid": len(diffs), "bootstrap_reps_requested": args.reps,
                        "block_keyframes": block, "pairs": len(keys),
                        "positive_pairs": labels[1], "negative_pairs": labels[0],
                        "sampling_unit": "first-frame keyframe anchor; circular moving-block bootstrap"
                    }
                    results.append(entry)
                    print(f"{det:11} {scope:20} {metric:27} block={block:2} "
                          f"delta={b-a:+.6f} CI=[{entry['ci95_lower']:+.6f}, {entry['ci95_upper']:+.6f}]", flush=True)
    target = out / "adaptive_v2_mutual_ap_paired_bootstrap.csv"
    with target.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    check = out / "adaptive_v2_ap_definition_validation.csv"
    with check.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(reproduced[0]))
        writer.writeheader()
        writer.writerows(reproduced)
    meta = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "fixed_source": str(base / "sequence6_mutual_feature_metrics.csv"),
        "adaptive_source": str(adap / "sequence6_mutual_feature_metrics.csv"),
        "reference_definition": "Binary COLMAP overlap_label; scores mutual_inliers_2px or mutual_fraction_2px",
        "ap_definition": "Non-interpolated precision weighted by increments in recall; tied scores grouped; equals sklearn average_precision_score",
        "pre_registered_comparisons": "No: Adaptive v2 thresholds were chosen after observing Sequence 06 frame statistics",
        "bootstrap": "Paired, circular moving blocks of first-frame keyframe anchors; shared resample counts across methods; dependencies via second-frame reuse remain",
        "reps": args.reps, "blocks": args.blocks, "seed": args.seed,
        "original_summaries_verified": True,
        "n_matched_records": len(fixed),
        "limitations": ["A single AQUALOC sequence", "COLMAP-derived overlap labels", "No SLAM ATE/trajectory results", "No across-sequence generalization", "Block lengths are sensitivity analyses, not proof of independence"]
    }
    (out / "adaptive_v2_mutual_ap_bootstrap_metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print("\nAP baseline definitions: PASS. Files:")
    print(target)
    print(check)

if __name__ == "__main__":
    main()
