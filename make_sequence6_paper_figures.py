import argparse
import math
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ============================================================
# Helpers
# ============================================================

def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def savefig(path: Path, dpi=220):
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close()


def first_existing(paths):
    for p in paths:
        if p.exists():
            return p
    return None


def read_csv_any(path: Path):
    return pd.read_csv(path, encoding="utf-8-sig")


def pick_column(columns, candidates):
    cols_lower = {c.lower(): c for c in columns}
    for cand in candidates:
        if cand.lower() in cols_lower:
            return cols_lower[cand.lower()]
    for c in columns:
        cl = c.lower()
        for cand in candidates:
            if cand.lower() in cl:
                return c
    return None


# ============================================================
# Sensor plots
# ============================================================

def load_timestamps_ns(csv_path: Path):
    df = read_csv_any(csv_path)
    ts = pd.to_numeric(df.iloc[:, 0], errors="coerce").dropna().to_numpy(dtype=np.int64)
    return ts


def intervals_ms(ts_ns):
    return np.diff(ts_ns) / 1e6


def plot_sensor_timing(root: Path, out_dir: Path):
    raw_data = root / "raw_data"
    files = {
        "Camera": raw_data / "img_sequence_6.csv",
        "Depth": raw_data / "depth_sequence_6.csv",
        "IMU": raw_data / "imu_sequence_6.csv",
        "Magnetometer": raw_data / "mag_sequence_6.csv",
    }

    available = {k: v for k, v in files.items() if v.exists()}
    if not available:
        print("[WARN] Sensor CSV files not found, skipping sensor timing figures.")
        return

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.ravel()

    summary_rows = []

    for ax, (name, path) in zip(axes, available.items()):
        ts = load_timestamps_ns(path)
        dt = intervals_ms(ts)
        ax.hist(dt, bins=50)
        ax.set_title(f"{name} interval histogram")
        ax.set_xlabel("Interval [ms]")
        ax.set_ylabel("Count")

        summary_rows.append({
            "sensor": name,
            "samples": len(ts),
            "duration_s": (ts[-1] - ts[0]) / 1e9 if len(ts) > 1 else 0,
            "mean_interval_ms": float(np.mean(dt)) if len(dt) else np.nan,
            "median_interval_ms": float(np.median(dt)) if len(dt) else np.nan,
            "min_interval_ms": float(np.min(dt)) if len(dt) else np.nan,
            "max_interval_ms": float(np.max(dt)) if len(dt) else np.nan,
        })

    savefig(out_dir / "figure_01_sensor_interval_histograms.png")

    if summary_rows:
        sdf = pd.DataFrame(summary_rows)
        plt.figure(figsize=(8, 5))
        x = np.arange(len(sdf))
        width = 0.35
        plt.bar(x - width/2, sdf["mean_interval_ms"], width, label="Mean")
        plt.bar(x + width/2, sdf["median_interval_ms"], width, label="Median")
        plt.xticks(x, sdf["sensor"])
        plt.ylabel("Interval [ms]")
        plt.title("Sensor timing summary")
        plt.legend()
        savefig(out_dir / "figure_02_sensor_timing_summary.png")

        sdf.to_csv(out_dir / "sensor_timing_summary.csv", index=False)


# ============================================================
# Trajectory / ATE plots
# ============================================================

def load_traj(path: Path):
    df = pd.read_csv(path, sep=r"[\s,]+", engine="python", comment="#", header=None)
    df = df.dropna(axis=1, how="all")

    if df.shape[1] < 8:
        raise ValueError(f"{path} has fewer than 8 columns.")

    df = df.iloc[:, :8].copy()
    df.columns = ["frame_id", "tx", "ty", "tz", "qx", "qy", "qz", "qw"]
    df["frame_id"] = df["frame_id"].astype(int)
    return df


def umeyama_alignment(src, dst, with_scale=True):
    # src, dst: Nx3
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)

    mu_src = src.mean(axis=0)
    mu_dst = dst.mean(axis=0)

    src_c = src - mu_src
    dst_c = dst - mu_dst

    cov = (dst_c.T @ src_c) / src.shape[0]
    U, D, Vt = np.linalg.svd(cov)

    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[-1, -1] = -1

    R = U @ S @ Vt

    if with_scale:
        var_src = np.mean(np.sum(src_c ** 2, axis=1))
        scale = np.trace(np.diag(D) @ S) / var_src
    else:
        scale = 1.0

    t = mu_dst - scale * (R @ mu_src)
    return scale, R, t


def apply_sim3(points, scale, R, t):
    return (scale * (R @ points.T)).T + t


def plot_trajectory_and_ate(root: Path, out_dir: Path):
    traj_a = first_existing([
        root / "colmap_traj_sequence_6.txt",
        root / "raw_data" / "colmap_traj_sequence_6.txt",
    ])
    traj_b = first_existing([
        root / "new_archaeo_colmap_traj_sequence_06.txt",
        root / "raw_data" / "new_archaeo_colmap_traj_sequence_06.txt",
    ])

    if traj_a is None or traj_b is None:
        print("[WARN] Trajectory files not found, skipping trajectory/ATE figures.")
        return

    A = load_traj(traj_a)
    B = load_traj(traj_b)

    merged = A.merge(B, on="frame_id", suffixes=("_a", "_b"))
    if len(merged) < 3:
        print("[WARN] Too few common poses, skipping trajectory/ATE figures.")
        return

    pa = merged[["tx_a", "ty_a", "tz_a"]].to_numpy()
    pb = merged[["tx_b", "ty_b", "tz_b"]].to_numpy()

    s, R, t = umeyama_alignment(pb, pa, with_scale=True)
    pb_aligned = apply_sim3(pb, s, R, t)
    ate = np.linalg.norm(pa - pb_aligned, axis=1)

    # XY/XZ/YZ overlays
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    proj = [("x", "y", 0, 1), ("x", "z", 0, 2), ("y", "z", 1, 2)]
    for ax, (lx, ly, i, j) in zip(axes, proj):
        ax.plot(pa[:, i], pa[:, j], label="Trajectory A")
        ax.plot(pb_aligned[:, i], pb_aligned[:, j], label="Trajectory B aligned")
        ax.set_xlabel(lx)
        ax.set_ylabel(ly)
        ax.set_title(f"{lx.upper()}-{ly.upper()} projection")
        ax.axis("equal")
        ax.legend()
    savefig(out_dir / "figure_03_trajectory_projections.png")

    # ATE histogram
    plt.figure(figsize=(7, 4))
    plt.hist(ate, bins=40)
    plt.xlabel("ATE [same units as trajectory]")
    plt.ylabel("Count")
    plt.title("ATE histogram after Sim(3) alignment")
    savefig(out_dir / "figure_04_ate_histogram.png")

    # ATE CDF
    xs = np.sort(ate)
    ys = np.arange(1, len(xs) + 1) / len(xs)
    plt.figure(figsize=(7, 4))
    plt.plot(xs, ys)
    plt.xlabel("ATE [same units as trajectory]")
    plt.ylabel("CDF")
    plt.title("ATE cumulative distribution")
    savefig(out_dir / "figure_05_ate_cdf.png")

    pd.DataFrame({
        "frame_id": merged["frame_id"],
        "ate": ate
    }).to_csv(out_dir / "trajectory_ate_values.csv", index=False)


# ============================================================
# Loop matrix
# ============================================================

def plot_loop_matrix(root: Path, out_dir: Path):
    loop_path = first_existing([
        root / "colmap_detected_loop_sequence_6.txt",
        root / "raw_data" / "colmap_detected_loop_sequence_6.txt",
    ])
    if loop_path is None:
        print("[WARN] Loop matrix not found, skipping.")
        return

    M = np.loadtxt(loop_path, delimiter=",")
    plt.figure(figsize=(6, 6))
    plt.imshow(M, aspect="auto")
    plt.title("COLMAP detected loop matrix - Sequence 06")
    plt.xlabel("Pose index")
    plt.ylabel("Pose index")
    plt.colorbar(label="Loop label")
    savefig(out_dir / "figure_06_loop_matrix.png")


# ============================================================
# Pair distribution plots
# ============================================================

def plot_pair_distributions(root: Path, out_dir: Path):
    manifest = root / "raw_data" / "sequence6_calibrated_reference" / "sequence6_calibrated_pair_manifest.csv"
    if not manifest.exists():
        print("[WARN] Pair manifest not found, skipping pair distribution plots.")
        return

    df = read_csv_any(manifest)

    fi = pick_column(df.columns, ["frame_id_i", "frame_i", "i_frame"])
    fj = pick_column(df.columns, ["frame_id_j", "frame_j", "j_frame"])
    label_col = pick_column(df.columns, ["overlap_label", "label", "positive"])
    base_col = pick_column(df.columns, ["baseline_m", "baseline", "baseline_distance"])

    if fi is None or fj is None or label_col is None:
        print("[WARN] Required columns not found in pair manifest, skipping.")
        return

    df["gap_frames"] = pd.to_numeric(df[fj]) - pd.to_numeric(df[fi])
    df["gap_steps"] = (df["gap_frames"] / 20.0).round().astype(int)
    df["label_num"] = pd.to_numeric(df[label_col]).astype(int)

    # Positive rate by keyframe gap
    gap_stats = df.groupby("gap_steps").agg(
        total=("label_num", "size"),
        positives=("label_num", "sum")
    ).reset_index()
    gap_stats["positive_rate"] = gap_stats["positives"] / gap_stats["total"]

    plt.figure(figsize=(8, 4.5))
    plt.plot(gap_stats["gap_steps"], gap_stats["positive_rate"], marker="o", linestyle="-")
    plt.xlabel("Keyframe gap")
    plt.ylabel("Positive overlap rate")
    plt.title("Positive overlap rate vs keyframe separation")
    savefig(out_dir / "figure_07_positive_rate_vs_keyframe_gap.png")

    gap_stats.to_csv(out_dir / "positive_rate_vs_keyframe_gap.csv", index=False)

    if base_col is not None:
        df["baseline_num"] = pd.to_numeric(df[base_col], errors="coerce")
        bins = [0, 0.05, 0.10, 0.25, 0.50, 1.00, np.inf]
        labels = ["<0.05", "0.05-0.10", "0.10-0.25", "0.25-0.50", "0.50-1.00", ">=1.00"]
        df["baseline_bin"] = pd.cut(df["baseline_num"], bins=bins, labels=labels, include_lowest=True, right=False)
        bstats = df.groupby("baseline_bin").agg(
            total=("label_num", "size"),
            positives=("label_num", "sum")
        ).reset_index()
        bstats["positive_rate"] = bstats["positives"] / bstats["total"]

        plt.figure(figsize=(8, 4.5))
        plt.bar(bstats["baseline_bin"].astype(str), bstats["positive_rate"])
        plt.xlabel("Baseline bin")
        plt.ylabel("Positive overlap rate")
        plt.title("Positive overlap rate vs baseline")
        savefig(out_dir / "figure_08_positive_rate_vs_baseline.png")

        bstats.to_csv(out_dir / "positive_rate_vs_baseline.csv", index=False)


# ============================================================
# Benchmark summary plots
# ============================================================

def operating_point_plot(summary_df, detector_order, prep_order, metric_col, out_path, title,
                         threshold_col="sampson_threshold_px",
                         threshold_value=2.0,
                         min_col=None,
                         min_value=8):
    df = summary_df.copy()
    if threshold_col in df.columns:
        df = df[np.isclose(pd.to_numeric(df[threshold_col]), threshold_value)]
    if min_col and min_col in df.columns:
        df = df[pd.to_numeric(df[min_col]) == min_value]
    if "scope" in df.columns:
        df = df[df["scope"] == "all_pairs"]

    rows = []
    for det in detector_order:
        for prep in prep_order:
            sel = df[(df["detector"] == det) & (df["preprocessing"].str.lower() == prep.lower())]
            if len(sel) == 0:
                rows.append({"detector": det, "preprocessing": prep, metric_col: np.nan})
            else:
                rows.append({"detector": det, "preprocessing": prep, metric_col: float(sel.iloc[0][metric_col])})

    plot_df = pd.DataFrame(rows)

    plt.figure(figsize=(10, 5))
    dets = detector_order
    preps = prep_order
    x = np.arange(len(dets))
    width = 0.24 if len(preps) >= 3 else 0.35

    for k, prep in enumerate(preps):
        vals = []
        for det in dets:
            row = plot_df[(plot_df["detector"] == det) & (plot_df["preprocessing"] == prep)]
            vals.append(float(row.iloc[0][metric_col]) if len(row) else np.nan)
        offset = (k - (len(preps)-1)/2) * width
        plt.bar(x + offset, vals, width=width, label=prep)

    plt.xticks(x, dets)
    plt.ylim(0, 1.05)
    plt.ylabel(metric_col)
    plt.title(title)
    plt.legend()
    savefig(out_path)


def heatmap_cost_maps(summary_df, detector_order, prep_order, metric_col, scope, out_path,
                      threshold_col="sampson_threshold_px", min_col=None):
    fig, axes = plt.subplots(len(detector_order), len(prep_order), figsize=(14, 10))
    if len(detector_order) == 1 and len(prep_order) == 1:
        axes = np.array([[axes]])
    elif len(detector_order) == 1:
        axes = np.array([axes])
    elif len(prep_order) == 1:
        axes = np.array([[a] for a in axes])

    for i, det in enumerate(detector_order):
        for j, prep in enumerate(prep_order):
            ax = axes[i, j]
            sub = summary_df[
                (summary_df["detector"] == det) &
                (summary_df["preprocessing"].str.lower() == prep.lower()) &
                (summary_df["scope"] == scope)
            ].copy()

            if len(sub) == 0:
                ax.set_title(f"{det} | {prep}\n(no data)")
                ax.axis("off")
                continue

            sub[threshold_col] = pd.to_numeric(sub[threshold_col])
            sub[min_col] = pd.to_numeric(sub[min_col])
            sub[metric_col] = pd.to_numeric(sub[metric_col])

            pv = sub.pivot(index=min_col, columns=threshold_col, values=metric_col)
            pv = pv.sort_index().sort_index(axis=1)

            im = ax.imshow(pv.values, aspect="auto", origin="lower")
            ax.set_title(f"{det} | {prep}")
            ax.set_xlabel("Sampson threshold [px]")
            ax.set_ylabel(min_col)
            ax.set_xticks(np.arange(len(pv.columns)))
            ax.set_xticklabels([str(c) for c in pv.columns])
            ax.set_yticks(np.arange(len(pv.index)))
            ax.set_yticklabels([str(r) for r in pv.index])

            for rr in range(pv.shape[0]):
                for cc in range(pv.shape[1]):
                    val = pv.values[rr, cc]
                    if np.isfinite(val):
                        ax.text(cc, rr, f"{val:.2f}", ha="center", va="center", fontsize=7)

            fig.colorbar(im, ax=ax, shrink=0.8)

    fig.suptitle(f"{metric_col} heatmaps | scope={scope}", y=1.01)
    savefig(out_path)


def plot_summary_benchmarks(root: Path, out_dir: Path):
    calibrated_summary = root / "raw_data" / "sequence6_calibrated_feature_results" / "sequence6_reference_feature_summary.csv"
    mutual_summary = root / "raw_data" / "sequence6_mutual_matching_results" / "sequence6_mutual_feature_summary.csv"
    adaptive_mutual_summary = root / "sequence6_adaptive_v2_mutual_matching_results" / "sequence6_mutual_feature_summary.csv"
    adaptive_cal_summary = root / "sequence6_adaptive_v2_calibrated_reference_results" / "sequence6_reference_feature_summary.csv"

    detectors = ["ORB", "AKAZE_MLDB", "SIFT"]
    preps_base = ["raw", "clahe"]
    preps_full = ["raw", "clahe", "adaptive_v2"]

    # Calibrated operating point
    if calibrated_summary.exists() and adaptive_cal_summary.exists():
        cfix = read_csv_any(calibrated_summary)
        cad = read_csv_any(adaptive_cal_summary)
        csum = pd.concat([cfix, cad], ignore_index=True)

        operating_point_plot(
            csum, detectors, preps_full, "f1",
            out_dir / "figure_09_calibrated_f1_operating_point.png",
            "Calibrated reference F1 | all_pairs | 2 px | min=8",
            threshold_col="sampson_threshold_px",
            threshold_value=2.0,
            min_col="minimum_reference_inliers",
            min_value=8
        )

        operating_point_plot(
            csum, detectors, preps_full, "average_precision_2px",
            out_dir / "figure_10_calibrated_ap_operating_point.png",
            "Calibrated reference AP | all_pairs",
            threshold_col="sampson_threshold_px",
            threshold_value=2.0,
            min_col="minimum_reference_inliers",
            min_value=8
        )

        heatmap_cost_maps(
            cfix, detectors, preps_base, "f1", "all_pairs",
            out_dir / "figure_11_calibrated_f1_cost_maps_all_pairs.png",
            threshold_col="sampson_threshold_px",
            min_col="minimum_reference_inliers"
        )

    # Mutual operating point
    if mutual_summary.exists() and adaptive_mutual_summary.exists():
        mfix = read_csv_any(mutual_summary)
        mad = read_csv_any(adaptive_mutual_summary)
        msum = pd.concat([mfix, mad], ignore_index=True)

        operating_point_plot(
            msum, detectors, preps_full, "f1",
            out_dir / "figure_12_mutual_f1_operating_point.png",
            "Mutual matching F1 | all_pairs | 2 px | min=8",
            threshold_col="sampson_threshold_px",
            threshold_value=2.0,
            min_col="minimum_mutual_inliers",
            min_value=8
        )

        # AP column may differ
        ap_col = "average_precision_count" if "average_precision_count" in msum.columns else "average_precision_fraction"
        operating_point_plot(
            msum, detectors, preps_full, ap_col,
            out_dir / "figure_13_mutual_ap_operating_point.png",
            f"Mutual matching {ap_col} | all_pairs",
            threshold_col="sampson_threshold_px",
            threshold_value=2.0,
            min_col="minimum_mutual_inliers",
            min_value=8
        )

        heatmap_cost_maps(
            mfix, detectors, preps_base, "f1", "all_pairs",
            out_dir / "figure_14_mutual_f1_cost_maps_all_pairs.png",
            threshold_col="sampson_threshold_px",
            min_col="minimum_mutual_inliers"
        )


# ============================================================
# Bootstrap CI plots
# ============================================================

def plot_bootstrap_cis(root: Path, out_dir: Path):
    paper_validation = root / "sequence6_paper_validation"

    f1_path = paper_validation / "adaptive_reference_f1_bootstrap.csv"
    ap_path = paper_validation / "adaptive_v2_mutual_ap_paired_bootstrap.csv"

    if f1_path.exists():
        df = read_csv_any(f1_path)

        # expect columns: detector, scope, difference, ci95_lower, ci95_upper
        if "difference" in df.columns:
            value_col = "difference"
        else:
            value_col = "paired_difference_adaptive_minus_fixed"

        scopes = ["all_pairs", "local_gap_le_100", "nonlocal_gap_ge_200"]
        dets = ["ORB", "AKAZE_MLDB", "SIFT"]
        xpos = np.arange(len(dets))
        width = 0.22

        plt.figure(figsize=(10, 5))
        for k, scope in enumerate(scopes):
            vals, lows, highs = [], [], []
            for det in dets:
                sub = df[(df["detector"] == det) & (df["scope"] == scope)]
                if len(sub) == 0:
                    vals.append(np.nan); lows.append(np.nan); highs.append(np.nan)
                else:
                    v = float(sub.iloc[0][value_col])
                    lo = float(sub.iloc[0]["ci95_lower"])
                    hi = float(sub.iloc[0]["ci95_upper"])
                    vals.append(v)
                    lows.append(v - lo)
                    highs.append(hi - v)
            plt.errorbar(
                xpos + (k-1)*width, vals, yerr=[lows, highs],
                fmt='o', capsize=4, label=scope
            )
        plt.axhline(0, linestyle="--")
        plt.xticks(xpos, dets)
        plt.ylabel("Adaptive - Fixed F1")
        plt.title("Bootstrap confidence intervals for F1 differences")
        plt.legend()
        savefig(out_dir / "figure_15_bootstrap_f1_confidence_intervals.png")

    if ap_path.exists():
        df = read_csv_any(ap_path)
        if "metric" in df.columns:
            df = df[df["metric"] == "average_precision_count"]

        dets = ["ORB", "AKAZE_MLDB", "SIFT"]
        scopes = ["all_pairs", "local_gap_le_100", "nonlocal_gap_ge_200"]
        xpos = np.arange(len(dets))
        width = 0.22

        plt.figure(figsize=(10, 5))
        for k, scope in enumerate(scopes):
            vals, lows, highs = [], [], []
            for det in dets:
                sub = df[(df["detector"] == det) & (df["scope"] == scope)]
                if len(sub) == 0:
                    vals.append(np.nan); lows.append(np.nan); highs.append(np.nan)
                else:
                    v = float(sub.iloc[0]["paired_difference_adaptive_minus_fixed"])
                    lo = float(sub.iloc[0]["ci95_lower"])
                    hi = float(sub.iloc[0]["ci95_upper"])
                    vals.append(v)
                    lows.append(v - lo)
                    highs.append(hi - v)
            plt.errorbar(
                xpos + (k-1)*width, vals, yerr=[lows, highs],
                fmt='o', capsize=4, label=scope
            )
        plt.axhline(0, linestyle="--")
        plt.xticks(xpos, dets)
        plt.ylabel("Adaptive - Fixed AP")
        plt.title("Bootstrap confidence intervals for AP differences")
        plt.legend()
        savefig(out_dir / "figure_16_bootstrap_ap_confidence_intervals.png")


# ============================================================
# Timing and adaptive decisions
# ============================================================

def plot_timing_and_decisions(root: Path, out_dir: Path):
    paper_validation = root / "sequence6_paper_validation"
    timing_summary = paper_validation / "preprocessing_timing_summary.csv"
    decisions_path = root / "sequence6_adaptive_v2_full" / "adaptive_frame_decisions.csv"

    if timing_summary.exists():
        df = read_csv_any(timing_summary)
        method_col = pick_column(df.columns, ["method"])
        mean_col = pick_column(df.columns, ["mean_ms"])
        if method_col and mean_col:
            plt.figure(figsize=(8, 4.5))
            plt.bar(df[method_col].astype(str), pd.to_numeric(df[mean_col]))
            plt.ylabel("Mean preprocessing time [ms]")
            plt.title("Preprocessing timing comparison")
            savefig(out_dir / "figure_17_preprocessing_timing.png")

    if decisions_path.exists():
        df = read_csv_any(decisions_path)

        frame_col = pick_column(df.columns, ["frame_id", "frame"])
        decision_col = None
        for c in df.columns:
            vals = set(df[c].astype(str).str.lower().unique())
            if any(v in vals for v in ["raw", "moderate_clahe", "strong_clahe"]):
                decision_col = c
                break

        if frame_col and decision_col:
            mapv = {"raw": 0, "moderate_clahe": 1, "strong_clahe": 2}
            xs = pd.to_numeric(df[frame_col])
            ys = df[decision_col].astype(str).str.lower().map(mapv)

            plt.figure(figsize=(12, 2.8))
            plt.plot(xs, ys, drawstyle="steps-mid")
            plt.yticks([0, 1, 2], ["raw", "moderate_clahe", "strong_clahe"])
            plt.xlabel("Frame ID")
            plt.ylabel("Decision")
            plt.title("Adaptive CLAHE v2 decisions over Sequence 06")
            savefig(out_dir / "figure_18_adaptive_decision_timeline.png")

            counts = df[decision_col].astype(str).str.lower().value_counts()
            plt.figure(figsize=(6, 4))
            plt.bar(counts.index, counts.values)
            plt.ylabel("Frame count")
            plt.title("Adaptive CLAHE decision distribution")
            savefig(out_dir / "figure_19_adaptive_decision_distribution.png")


# ============================================================
# Feature maps and match visualization
# ============================================================

def apply_clahe(gray, clip_limit=2.0, tile_grid=(8, 8)):
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid)
    return clahe.apply(gray)


def load_decision_for_frame(decision_csv: Path, frame_id: int):
    if not decision_csv.exists():
        return "moderate_clahe"

    df = read_csv_any(decision_csv)
    frame_col = pick_column(df.columns, ["frame_id", "frame"])
    decision_col = None
    for c in df.columns:
        vals = set(df[c].astype(str).str.lower().unique())
        if any(v in vals for v in ["raw", "moderate_clahe", "strong_clahe"]):
            decision_col = c
            break

    if frame_col is None or decision_col is None:
        return "moderate_clahe"

    sub = df[pd.to_numeric(df[frame_col], errors="coerce") == frame_id]
    if len(sub) == 0:
        return "moderate_clahe"
    return str(sub.iloc[0][decision_col]).lower()


def get_detector(name):
    name = name.upper()
    if name == "ORB":
        return cv2.ORB_create(nfeatures=1500)
    elif name == "AKAZE_MLDB":
        return cv2.AKAZE_create(descriptor_type=cv2.AKAZE_DESCRIPTOR_MLDB)
    elif name == "SIFT":
        return cv2.SIFT_create(nfeatures=1500)
    else:
        raise ValueError(name)


def detect_and_draw(gray, detector_name):
    det = get_detector(detector_name)
    kps, des = det.detectAndCompute(gray, None)
    out = cv2.drawKeypoints(gray, kps, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    return out, kps, des


def preprocess_variant(gray, variant, decision):
    if variant == "raw":
        return gray
    elif variant == "clahe":
        return apply_clahe(gray)
    elif variant == "adaptive_v2":
        if decision == "raw":
            return gray
        else:
            return apply_clahe(gray)
    else:
        raise ValueError(variant)


def plot_feature_maps(root: Path, out_dir: Path, example_frame=100):
    images_dir = root / "raw_data" / "images_sequence_6"
    decision_csv = root / "sequence6_adaptive_v2_full" / "adaptive_frame_decisions.csv"

    if not images_dir.exists():
        print("[WARN] images_sequence_6 folder not found, skipping feature maps.")
        return

    img_path = images_dir / f"frame{example_frame:06d}.png"
    if not img_path.exists():
        img_path = images_dir / "frame000000.png"
        example_frame = 0
    if not img_path.exists():
        print("[WARN] Example image not found, skipping feature maps.")
        return

    gray = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        print("[WARN] Could not read example image, skipping feature maps.")
        return

    decision = load_decision_for_frame(decision_csv, example_frame)

    detectors = ["ORB", "AKAZE_MLDB", "SIFT"]
    variants = ["raw", "clahe", "adaptive_v2"]

    fig, axes = plt.subplots(len(detectors), len(variants), figsize=(14, 12))

    for i, det_name in enumerate(detectors):
        for j, variant in enumerate(variants):
            proc = preprocess_variant(gray, variant, decision)
            vis, _, _ = detect_and_draw(proc, det_name)
            ax = axes[i, j]
            ax.imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
            ax.axis("off")
            ax.set_title(f"{det_name} | {variant}")

    fig.suptitle(f"Feature maps on frame {example_frame} (adaptive decision={decision})", y=1.01)
    savefig(out_dir / "figure_20_feature_map_grid.png")


def ratio_match(des1, des2, detector_name, ratio=0.75):
    if des1 is None or des2 is None:
        return []

    if detector_name in ["ORB", "AKAZE_MLDB"]:
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    else:
        matcher = cv2.BFMatcher(cv2.NORM_L2)

    knn = matcher.knnMatch(des1, des2, k=2)
    good = []
    for pair in knn:
        if len(pair) < 2:
            continue
        m, n = pair
        if m.distance < ratio * n.distance:
            good.append(m)
    return good


def plot_match_visualization(root: Path, out_dir: Path, frame_a=0, frame_b=20, detector_name="ORB"):
    images_dir = root / "raw_data" / "images_sequence_6"
    decision_csv = root / "sequence6_adaptive_v2_full" / "adaptive_frame_decisions.csv"

    img_a = images_dir / f"frame{frame_a:06d}.png"
    img_b = images_dir / f"frame{frame_b:06d}.png"
    if not img_a.exists() or not img_b.exists():
        print("[WARN] Match visualization frames not found, skipping.")
        return

    ga = cv2.imread(str(img_a), cv2.IMREAD_GRAYSCALE)
    gb = cv2.imread(str(img_b), cv2.IMREAD_GRAYSCALE)

    dec_a = load_decision_for_frame(decision_csv, frame_a)
    dec_b = load_decision_for_frame(decision_csv, frame_b)

    pa = preprocess_variant(ga, "adaptive_v2", dec_a)
    pb = preprocess_variant(gb, "adaptive_v2", dec_b)

    det = get_detector(detector_name)
    k1, d1 = det.detectAndCompute(pa, None)
    k2, d2 = det.detectAndCompute(pb, None)
    good = ratio_match(d1, d2, detector_name, ratio=0.75)
    good = sorted(good, key=lambda m: m.distance)[:80]

    vis = cv2.drawMatches(pa, k1, pb, k2, good, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    plt.figure(figsize=(16, 8))
    plt.imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.title(f"Adaptive v2 match visualization | {detector_name} | frame {frame_a} vs {frame_b}")
    savefig(out_dir / "figure_21_match_visualization.png")


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, help="Path to archaeo_sequence_6_raw_data")
    parser.add_argument("--out", default=None, help="Output folder for PNG figures")
    parser.add_argument("--example-frame", type=int, default=100)
    parser.add_argument("--match-a", type=int, default=0)
    parser.add_argument("--match-b", type=int, default=20)
    args = parser.parse_args()

    root = Path(args.root)
    out_dir = Path(args.out) if args.out else root / "paper_figures_png"
    ensure_dir(out_dir)

    print("[INFO] Output directory:", out_dir)

    plot_sensor_timing(root, out_dir)
    print("[DONE] Sensor plots")

    plot_trajectory_and_ate(root, out_dir)
    print("[DONE] Trajectory / ATE plots")

    plot_loop_matrix(root, out_dir)
    print("[DONE] Loop matrix")

    plot_pair_distributions(root, out_dir)
    print("[DONE] Pair distributions")

    plot_summary_benchmarks(root, out_dir)
    print("[DONE] Benchmark summary plots")

    plot_bootstrap_cis(root, out_dir)
    print("[DONE] Bootstrap confidence interval plots")

    plot_timing_and_decisions(root, out_dir)
    print("[DONE] Timing / adaptive decision plots")

    plot_feature_maps(root, out_dir, example_frame=args.example_frame)
    print("[DONE] Feature maps")

    plot_match_visualization(root, out_dir, frame_a=args.match_a, frame_b=args.match_b, detector_name="ORB")
    print("[DONE] Match visualization")

    print("[FINISHED] Figures saved in:", out_dir)


if __name__ == "__main__":
    main()