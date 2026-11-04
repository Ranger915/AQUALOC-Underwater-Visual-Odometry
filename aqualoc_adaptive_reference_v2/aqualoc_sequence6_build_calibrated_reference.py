import csv
import json
import math
import re
from pathlib import Path

import numpy as np

ROOT = Path.home() / "Downloads" / "archaeo_sequence_6_raw_data"

TRAJECTORY = ROOT / "colmap_traj_sequence_6.txt"
LOOP_FILE = ROOT / "colmap_detected_loop_sequence_6.txt"

OUT = ROOT / "raw_data" / "sequence6_calibrated_reference"
OUT.mkdir(parents=True, exist_ok=True)

# Official AQUALOC archaeological camera calibration
FX = 543.3327734182214
FY = 542.398772982566
CX = 489.02536042247897
CY = 305.38727712002805

DISTORTION = [
    -0.1255945656257394,
     0.053221287232781606,
     9.94070021080493e-05,
     9.550660927242349e-05
]

K = np.array([
    [FX, 0.0, CX],
    [0.0, FY, CY],
    [0.0, 0.0, 1.0]
], dtype=float)

K_INV = np.linalg.inv(K)


def read_numeric_rows(path):
    if not path.is_file():
        raise FileNotFoundError(path)

    rows = []

    with open(path, "r", encoding="utf-8-sig") as f:
        for line_number, line in enumerate(f, 1):
            line = line.strip()

            if not line or line.startswith("#"):
                continue

            tokens = re.split(r"[\s,]+", line)

            try:
                values = [float(x) for x in tokens if x]
            except ValueError:
                continue

            if values:
                rows.append((line_number, values))

    return rows


def read_trajectory(path):
    parsed = read_numeric_rows(path)
    poses = []

    for line_number, values in parsed:
        if len(values) != 8:
            raise ValueError(
                f"Expected 8 trajectory values at line {line_number}, "
                f"found {len(values)}"
            )

        frame_float = values[0]
        frame_id = int(round(frame_float))

        if abs(frame_float - frame_id) > 1e-8:
            raise ValueError(f"Invalid frame ID at line {line_number}")

        q = np.asarray(values[4:8], dtype=float)
        q_norm = np.linalg.norm(q)

        if abs(q_norm - 1.0) > 1e-6:
            raise ValueError(
                f"Quaternion norm problem at frame {frame_id}: {q_norm}"
            )

        poses.append({
            "frame_id": frame_id,
            "C": np.asarray(values[1:4], dtype=float),
            "q_xyzw": q / q_norm,
            "q_norm": q_norm
        })

    ids = [p["frame_id"] for p in poses]

    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate frame IDs found")

    if ids != sorted(ids):
        raise ValueError("Trajectory frame IDs are not increasing")

    return poses


def read_loop_matrix(path):
    rows = read_numeric_rows(path)
    matrix_rows = [values for _, values in rows]

    lengths = {len(row) for row in matrix_rows}

    if len(lengths) != 1:
        raise ValueError("Loop matrix has inconsistent row lengths")

    M = np.asarray(matrix_rows, dtype=float)

    if M.ndim != 2 or M.shape[0] != M.shape[1]:
        raise ValueError(f"Loop matrix is not square: {M.shape}")

    if not np.all(np.isin(M, [0.0, 1.0])):
        raise ValueError("Loop matrix contains values other than 0 and 1")

    return M.astype(int)


def quaternion_xyzw_to_rotation(q):
    x, y, z, w = q

    return np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - z*w),     2*(x*z + y*w)],
        [2*(x*y + z*w),     1 - 2*(x*x + z*z), 2*(y*z - x*w)],
        [2*(x*z - y*w),     2*(y*z + x*w),     1 - 2*(x*x + y*y)]
    ], dtype=float)


def skew(v):
    x, y, z = v

    return np.array([
        [0.0, -z, y],
        [z, 0.0, -x],
        [-y, x, 0.0]
    ], dtype=float)


def rotation_angle_degrees(R):
    c = (np.trace(R) - 1.0) / 2.0
    c = np.clip(c, -1.0, 1.0)
    return math.degrees(math.acos(c))


def write_csv(path, header, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


def main():
    print("=" * 76)
    print("AQUALOC SEQUENCE 06 - CALIBRATED GEOMETRY REFERENCE")
    print("=" * 76)

    poses = read_trajectory(TRAJECTORY)
    M = read_loop_matrix(LOOP_FILE)

    n = len(poses)

    print("\n1. INPUT VALIDATION")
    print("-" * 76)
    print("Trajectory poses:", n)
    print("Loop matrix shape:", M.shape)

    if M.shape != (n, n):
        raise ValueError(
            "Loop matrix dimensions do not match trajectory pose count"
        )

    if not np.array_equal(M, M.T):
        raise ValueError("Loop matrix is not symmetric")

    if not np.all(np.diag(M) == 1):
        raise ValueError("Loop matrix diagonal is not all ones")

    frame_ids = [p["frame_id"] for p in poses]

    gaps = np.diff(frame_ids)

    if not np.all(gaps == 20):
        raise ValueError(
            "Trajectory is not sampled at a constant 20-frame interval"
        )

    print("Frame ID range:", frame_ids[0], "to", frame_ids[-1])
    print("Frame ID interval: 20")
    print("Matrix symmetric: YES")
    print("Binary labels: YES")
    print("All diagonal values equal 1: YES")

    rotations = [
        quaternion_xyzw_to_rotation(p["q_xyzw"])
        for p in poses
    ]

    for index, R in enumerate(rotations):
        if abs(np.linalg.det(R) - 1.0) > 1e-8:
            raise ValueError(f"Invalid rotation at pose index {index}")

        if np.linalg.norm(R.T @ R - np.eye(3)) > 1e-8:
            raise ValueError(f"Non-orthogonal rotation at pose index {index}")

    print("Rotation matrix checks: PASSED")

    # Save the canonical pose table.
    pose_header = [
        "pose_index", "frame_id", "image_filename",
        "tx", "ty", "tz", "qx", "qy", "qz", "qw",
        "quaternion_norm"
    ]

    pose_rows = []

    for i, p in enumerate(poses):
        pose_rows.append([
            i,
            p["frame_id"],
            f"frame{p['frame_id']:06d}.png",
            *p["C"].tolist(),
            *p["q_xyzw"].tolist(),
            p["q_norm"]
        ])

    pose_path = OUT / "sequence6_pose_reference.csv"
    write_csv(pose_path, pose_header, pose_rows)

    # Relative pose convention:
    # T_wc = [R_wc, C_w]
    # T_cj_ci = inverse(T_wc_j) * T_wc_i
    #
    # X_cj = R_ji * X_ci + t_ji

    pair_header = [
        "pose_index_i",
        "pose_index_j",
        "frame_id_i",
        "frame_id_j",
        "image_i",
        "image_j",
        "frame_gap",
        "colmap_overlap_label",
        "baseline_norm_export_units",
        "relative_rotation_degrees",
        "t_ji_x",
        "t_ji_y",
        "t_ji_z"
    ]

    for prefix in ("Rji", "E", "F"):
        pair_header.extend([
            f"{prefix}_{r}{c}"
            for r in range(3)
            for c in range(3)
        ])

    pair_rows = []

    overlap_positive = 0
    baseline_values = []
    rotation_values = []

    for i in range(n):
        for j in range(i + 1, n):

            pi = poses[i]
            pj = poses[j]

            Ri = rotations[i]
            Rj = rotations[j]

            Ci = pi["C"]
            Cj = pj["C"]

            # Camera i coordinates -> camera j coordinates
            Rji = Rj.T @ Ri
            tji = Rj.T @ (Ci - Cj)

            baseline = float(np.linalg.norm(Cj - Ci))
            angle = rotation_angle_degrees(Rji)

            E = skew(tji) @ Rji

            # F is defined for undistorted pixel coordinates.
            F = K_INV.T @ E @ K_INV

            f_norm = np.linalg.norm(F)

            if f_norm > 1e-15:
                F = F / f_norm

                # Deterministic sign convention
                largest = np.unravel_index(
                    np.argmax(np.abs(F)), F.shape
                )

                if F[largest] < 0:
                    F = -F

            else:
                raise ValueError(
                    f"Degenerate epipolar matrix for pair {i}, {j}"
                )

            overlap = int(M[i, j])

            overlap_positive += overlap
            baseline_values.append(baseline)
            rotation_values.append(angle)

            row = [
                i,
                j,
                pi["frame_id"],
                pj["frame_id"],
                f"frame{pi['frame_id']:06d}.png",
                f"frame{pj['frame_id']:06d}.png",
                pj["frame_id"] - pi["frame_id"],
                overlap,
                baseline,
                angle,
                *tji.tolist()
            ]

            row.extend(Rji.flatten().tolist())
            row.extend(E.flatten().tolist())
            row.extend(F.flatten().tolist())

            pair_rows.append(row)

    pair_path = OUT / "sequence6_calibrated_pair_manifest.csv"
    write_csv(pair_path, pair_header, pair_rows)

    total_pairs = len(pair_rows)
    overlap_negative = total_pairs - overlap_positive

    # Calibration metadata and formula conventions
    metadata = {
        "dataset": "AQUALOC Sequence 06",
        "reference_trajectory": TRAJECTORY.name,
        "loop_matrix": LOOP_FILE.name,
        "pose_count": n,
        "unique_unordered_pairs": total_pairs,
        "frame_sampling_interval": 20,
        "frame_mapping": "pose index k maps to trajectory frame ID at row k",
        "quaternion_order": "xyzw",
        "pose_convention": "camera-to-world",
        "relative_pose_convention": "X_cj = Rji * X_ci + tji",
        "essential_matrix": "E = skew(tji) * Rji",
        "fundamental_matrix": "F = inverse(K).T * E * inverse(K)",
        "fundamental_matrix_coordinates": "undistorted pixel coordinates",
        "undistortion_model": "RadTan",
        "camera_matrix": K.tolist(),
        "distortion_coefficients": DISTORTION,
        "image_resolution": [968, 608],
        "colmap_overlap_positive_pairs": overlap_positive,
        "colmap_overlap_negative_pairs": overlap_negative,
        "baseline_units": "trajectory export units; metric provenance should be confirmed",
        "warning": (
            "COLMAP overlap labels indicate image overlap, not independently "
            "verified online loop closures. Exclude temporal neighbours "
            "according to a separately declared evaluation protocol."
        )
    }

    metadata_path = OUT / "sequence6_calibrated_reference_metadata.json"

    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)

    print("\n2. PAIR MANIFEST")
    print("-" * 76)
    print("Unique unordered pairs:", total_pairs)
    print("Overlap-positive pairs:", overlap_positive)
    print("Overlap-negative pairs:", overlap_negative)
    print("Expected pairs:", n * (n - 1) // 2)

    print("\n3. RELATIVE GEOMETRY")
    print("-" * 76)
    print("Baseline min:", min(baseline_values))
    print("Baseline median:", float(np.median(baseline_values)))
    print("Baseline max:", max(baseline_values))
    print("Relative rotation median:", float(np.median(rotation_values)))
    print("Relative rotation max:", max(rotation_values))

    print("\n4. OUTPUT FILES")
    print("-" * 76)

    for path in (pose_path, pair_path, metadata_path):
        print(path)
        print("Size:", path.stat().st_size, "bytes")

    print("\n5. VALIDATION")
    print("-" * 76)
    print("Pose table rows:", len(pose_rows))
    print("Pair manifest rows:", len(pair_rows))
    print("Pose-to-loop mapping: VERIFIED BY ORDER AND FRAME SPACING")
    print("Camera calibration: OFFICIAL AQUALOC PARAMETERS")
    print("Original benchmark files modified: NO")

    print("\nOutput directory:", OUT)
    print("=" * 76)


if __name__ == "__main__":
    main()
