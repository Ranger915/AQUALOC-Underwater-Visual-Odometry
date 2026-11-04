import os
import re
import numpy as np

ROOT = os.path.join(
    os.path.expanduser("~"),
    "Downloads",
    "archaeo_sequence_6_raw_data"
)

FILES = {
    "trajectory_1": os.path.join(
        ROOT, "colmap_traj_sequence_6.txt"
    ),
    "trajectory_2": os.path.join(
        ROOT, "new_archaeo_colmap_traj_sequence_06.txt"
    ),
    "loop_matrix": os.path.join(
        ROOT, "colmap_detected_loop_sequence_6.txt"
    )
}


def read_numeric_file(path):
    if not os.path.isfile(path):
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
    raw = read_numeric_file(path)
    result = {}

    for line_number, values in raw:
        if len(values) != 8:
            raise ValueError(
                f"{os.path.basename(path)}: expected 8 values "
                f"at line {line_number}, found {len(values)}: {values}"
            )

        frame_id = int(round(values[0]))

        if abs(values[0] - frame_id) > 1e-8:
            raise ValueError(
                f"Non-integer frame ID at line {line_number}"
            )

        if frame_id in result:
            raise ValueError(f"Duplicate frame ID: {frame_id}")

        result[frame_id] = {
            "translation": np.array(values[1:4], dtype=float),
            "quaternion": np.array(values[4:8], dtype=float)
        }

    return result


def quaternion_to_rotation(q, order):
    q = np.asarray(q, dtype=float)

    if order == "wxyz":
        w, x, y, z = q
    elif order == "xyzw":
        x, y, z, w = q
    else:
        raise ValueError(order)

    norm = np.linalg.norm([w, x, y, z])

    if norm < 1e-12:
        raise ValueError("Zero quaternion")

    w, x, y, z = np.array([w, x, y, z]) / norm

    return np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - z*w),     2*(x*z + y*w)],
        [2*(x*y + z*w),     1 - 2*(x*x + z*z), 2*(y*z - x*w)],
        [2*(x*z - y*w),     2*(y*z + x*w),     1 - 2*(x*x + y*y)]
    ])


def rotation_error_degrees(R1, R2):
    relative = R1.T @ R2
    cosine = (np.trace(relative) - 1.0) / 2.0
    cosine = np.clip(cosine, -1.0, 1.0)
    return np.degrees(np.arccos(cosine))


def estimate_similarity(source, target):
    """
    Estimate target ~= scale * R * source + translation.
    Coordinates are represented as row vectors in the input arrays.
    R returned here acts on column vectors.
    """
    source_mean = source.mean(axis=0)
    target_mean = target.mean(axis=0)

    X = source - source_mean
    Y = target - target_mean

    covariance = X.T @ Y
    U, singular_values, Vt = np.linalg.svd(covariance)

    correction = np.eye(3)
    if np.linalg.det(U @ Vt) < 0:
        correction[2, 2] = -1

    R_row = U @ correction @ Vt
    scale = np.sum(singular_values * np.diag(correction)) / np.sum(X * X)

    R_column = R_row.T
    translation = target_mean - scale * (source_mean @ R_row)

    aligned = scale * (source @ R_row) + translation
    residuals = np.linalg.norm(aligned - target, axis=1)

    return scale, R_column, translation, residuals


def main():
    print("=" * 76)
    print("AQUALOC SEQUENCE 06 - COLMAP POSE CONVENTION AUDIT")
    print("=" * 76)

    for name, path in FILES.items():
        print(f"\n{name}: {path}")
        if not os.path.isfile(path):
            raise FileNotFoundError(path)
        print(f"Size: {os.path.getsize(path)} bytes")

    traj1 = read_trajectory(FILES["trajectory_1"])
    traj2 = read_trajectory(FILES["trajectory_2"])

    ids1 = list(traj1.keys())
    ids2 = list(traj2.keys())
    common = sorted(set(ids1) & set(ids2))

    print("\n1. TRAJECTORY STRUCTURE")
    print("-" * 76)

    for name, ids in [("Trajectory 1", ids1), ("Trajectory 2", ids2)]:
        intervals = np.diff(ids)
        print(f"\n{name}")
        print("Pose count:", len(ids))
        print("First 12 frame IDs:", ids[:12])
        print("Last 12 frame IDs:", ids[-12:])
        print("Strictly increasing:", all(a < b for a, b in zip(ids, ids[1:])))
        print("First raw pose:", ids[0], traj1[ids[0]] if name == "Trajectory 1" else traj2[ids[0]])

        if len(intervals):
            unique, counts = np.unique(intervals, return_counts=True)
            print("Frame-ID interval counts:", dict(zip(unique.tolist(), counts.tolist())))

    print("\nCommon frame IDs:", len(common))
    print("Only in trajectory 1:", sorted(set(ids1) - set(ids2)))
    print("Only in trajectory 2:", sorted(set(ids2) - set(ids1)))

    if len(common) < 3:
        raise RuntimeError("Insufficient common poses for alignment")

    source = np.array([traj2[i]["translation"] for i in common])
    target = np.array([traj1[i]["translation"] for i in common])

    scale, R_align, translation, residuals = estimate_similarity(
        source, target
    )

    print("\n2. POSITION-BASED SIMILARITY ALIGNMENT")
    print("-" * 76)
    print("Source: trajectory 2")
    print("Target: trajectory 1")
    print("Scale:", scale)
    print("Rotation matrix mapping source world to target world:")
    print(R_align)
    print("Translation:", translation)
    print("Rotation determinant:", np.linalg.det(R_align))
    print("Orthogonality error:",
          np.linalg.norm(R_align.T @ R_align - np.eye(3)))
    print("Position RMSE:", np.sqrt(np.mean(residuals**2)))
    print("Position median:", np.median(residuals))
    print("Position P95:", np.percentile(residuals, 95))
    print("Position maximum:", np.max(residuals))

    print("\n3. RAW QUATERNION DATA")
    print("-" * 76)

    for name, data in [("Trajectory 1", traj1), ("Trajectory 2", traj2)]:
        norms = np.array([
            np.linalg.norm(data[i]["quaternion"]) for i in data
        ])
        first_id = next(iter(data))
        print(f"\n{name}")
        print("First raw quaternion:", data[first_id]["quaternion"])
        print("Quaternion norm min:", norms.min())
        print("Quaternion norm max:", norms.max())
        print("Quaternion norm mean:", norms.mean())

    print("\n4. ORIENTATION-CONSISTENCY TEST")
    print("-" * 76)
    print("The position alignment above is used to test both pose conventions.")
    print("C2W: R_target ~= R_align @ R_source")
    print("W2C: R_target ~= R_source @ R_align.T")

    scores = []

    for order1 in ("wxyz", "xyzw"):
        for order2 in ("wxyz", "xyzw"):
            R_target_list = [
                quaternion_to_rotation(traj1[i]["quaternion"], order1)
                for i in common
            ]
            R_source_list = [
                quaternion_to_rotation(traj2[i]["quaternion"], order2)
                for i in common
            ]

            for convention in ("C2W", "W2C"):
                errors = []

                for Rt, Rs in zip(R_target_list, R_source_list):
                    if convention == "C2W":
                        predicted = R_align @ Rs
                    else:
                        predicted = Rs @ R_align.T

                    errors.append(rotation_error_degrees(predicted, Rt))

                errors = np.asarray(errors)

                scores.append((
                    float(np.median(errors)),
                    order1,
                    order2,
                    convention,
                    float(np.mean(errors)),
                    float(np.percentile(errors, 95)),
                    float(np.max(errors))
                ))

    scores.sort(key=lambda item: item[0])

    print("\nRanked convention candidates:")
    print("Quaternion order T1 | T2 | Pose convention | Median deg | Mean deg | P95 deg | Max deg")

    for score in scores:
        median, order1, order2, convention, mean, p95, maximum = score
        print(
            f"{order1:>5} | {order2:>5} | {convention:>4} | "
            f"{median:10.6f} | {mean:9.6f} | "
            f"{p95:9.6f} | {maximum:9.6f}"
        )

    print("\nBest mathematical candidate:", scores[0])

    print("\n5. LOOP MATRIX AUDIT")
    print("-" * 76)

    loop_rows = read_numeric_file(FILES["loop_matrix"])
    matrix_data = [row for _, row in loop_rows]

    lengths = [len(row) for row in matrix_data]

    print("Matrix rows:", len(matrix_data))
    print("Row lengths (unique):", sorted(set(lengths)))

    if len(set(lengths)) == 1:
        M = np.asarray(matrix_data, dtype=float)

        print("Matrix shape:", M.shape)
        print("Values limited to 0 and 1:",
              bool(np.all(np.isin(M, [0.0, 1.0]))))

        if M.ndim == 2 and M.shape[0] == M.shape[1]:
            print("Diagonal ones:", int(np.trace(M)))
            print("Positive entries:", int(np.sum(M)))
            print("Symmetric:", bool(np.array_equal(M, M.T)))
            print("Asymmetric off-diagonal pairs:",
                  int(np.sum(M != M.T) // 2))
            print("Row-sum min/max/mean:",
                  int(M.sum(axis=1).min()),
                  int(M.sum(axis=1).max()),
                  float(M.sum(axis=1).mean()))

            print("\nLoop index -> candidate trajectory 1 frame mapping")
            if M.shape[0] == len(ids1):
                for idx in list(range(min(10, len(ids1)))) + list(
                    range(max(10, len(ids1)-5), len(ids1))
                ):
                    print(f"{idx} -> frame {ids1[idx]}")

                print(
                    "\nDimension match: YES. The matrix has one row/column "
                    "per trajectory 1 pose."
                )
                print(
                    "Important: dimensions alone do not prove that the "
                    "matrix index order equals trajectory file order."
                )
            else:
                print("Dimension match with trajectory 1: NO")
        else:
            print("Loop matrix is not square.")
    else:
        print("Inconsistent row lengths; matrix parsing needs inspection.")

    print("\n6. AUDIT CONCLUSION")
    print("-" * 76)
    print("This script ranks mathematical convention candidates.")
    print("It does not independently prove the original file's semantic convention.")
    print("Use the source export documentation to confirm the winning candidate.")
    print("Do not calculate the fundamental matrix until pose semantics are confirmed.")
    print("=" * 76)


if __name__ == "__main__":
    main()
