import os
import math
from collections import Counter

ROOT = os.path.join(
    os.path.expanduser("~"),
    "Downloads",
    "archaeo_sequence_6_raw_data"
)

FILES = {
    "trajectory_1": os.path.join(ROOT, "colmap_traj_sequence_6.txt"),
    "trajectory_2": os.path.join(ROOT, "new_archaeo_colmap_traj_sequence_06.txt"),
    "loop_matrix": os.path.join(ROOT, "colmap_detected_loop_sequence_6.txt")
}


def load_trajectory(path):
    data = {}
    duplicates = []
    malformed = []

    if not os.path.isfile(path):
        raise FileNotFoundError(path)

    with open(path, "r", encoding="utf-8-sig") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue

            try:
                values = [float(x) for x in line.split()]

                if len(values) != 8:
                    malformed.append((line_no, "Expected 8 values"))
                    continue

                if not all(math.isfinite(x) for x in values):
                    malformed.append((line_no, "Non-finite value"))
                    continue

                if not values[0].is_integer():
                    malformed.append((line_no, "Non-integer frame ID"))
                    continue

                frame_id = int(values[0])

                if frame_id in data:
                    duplicates.append(frame_id)

                data[frame_id] = values[1:]

            except Exception as exc:
                malformed.append((line_no, str(exc)))

    return data, duplicates, malformed


def audit_trajectory(name, path):
    print("\n" + "=" * 75)
    print(name)
    print("=" * 75)

    data, duplicates, malformed = load_trajectory(path)
    ids = sorted(data)

    print("File:", path)
    print("Valid trajectory entries:", len(ids))
    print("Duplicate frame IDs:", len(duplicates))
    print("Malformed rows:", len(malformed))

    if duplicates:
        print("Duplicate IDs:", duplicates[:20])

    if malformed:
        print("Malformed row examples:", malformed[:10])

    if not ids:
        print("No valid trajectory entries.")
        return data

    print("Minimum frame ID:", ids[0])
    print("Maximum frame ID:", ids[-1])

    missing = sorted(set(range(ids[0], ids[-1] + 1)) - set(ids))
    print("Missing IDs within range:", len(missing))
    print("First missing IDs:", missing[:30])

    intervals = Counter(
        ids[i + 1] - ids[i] for i in range(len(ids) - 1)
    )
    print("Frame ID interval distribution:", dict(sorted(intervals.items())))

    norms = []
    for values in data.values():
        q = values[3:7]
        norms.append(math.sqrt(sum(x * x for x in q)))

    if norms:
        print("Quaternion norm min:", min(norms))
        print("Quaternion norm max:", max(norms))
        print("Mean quaternion norm:", sum(norms) / len(norms))
        print(
            "Quaternions near unit norm (0.99-1.01):",
            sum(0.99 <= n <= 1.01 for n in norms),
            "/", len(norms)
        )

    return data


def quaternion_angle_degrees(q1, q2):
    """
    Sign-invariant angular difference between two unit quaternions.
    Component ordering must be identical in both input trajectories.
    """
    n1 = math.sqrt(sum(x * x for x in q1))
    n2 = math.sqrt(sum(x * x for x in q2))

    if n1 == 0 or n2 == 0:
        return float("nan")

    q1 = [x / n1 for x in q1]
    q2 = [x / n2 for x in q2]

    dot = abs(sum(q1[i] * q2[i] for i in range(4)))
    dot = max(-1.0, min(1.0, dot))

    return math.degrees(2.0 * math.acos(dot))


print("=" * 75)
print("AQUALOC SEQUENCE 06 - COLMAP REFERENCE AUDIT")
print("=" * 75)

t1 = audit_trajectory("COLMAP TRAJECTORY 1", FILES["trajectory_1"])
t2 = audit_trajectory("COLMAP TRAJECTORY 2", FILES["trajectory_2"])


# ------------------------------------------------------------
# TRAJECTORY CROSS-COMPARISON
# ------------------------------------------------------------

print("\n" + "=" * 75)
print("TRAJECTORY CROSS-COMPARISON")
print("=" * 75)

common = sorted(set(t1) & set(t2))
only1 = sorted(set(t1) - set(t2))
only2 = sorted(set(t2) - set(t1))

print("Trajectory 1 entries:", len(t1))
print("Trajectory 2 entries:", len(t2))
print("Common frame IDs:", len(common))
print("Only in trajectory 1:", len(only1))
print("Only in trajectory 2:", len(only2))
print("First common IDs:", common[:25])
print("First IDs only in trajectory 1:", only1[:25])
print("First IDs only in trajectory 2:", only2[:25])

if common:
    translation_distances = []
    rotation_angles = []

    for fid in common:
        a = t1[fid]
        b = t2[fid]

        # Translation components
        ta = a[:3]
        tb = b[:3]

        distance = math.sqrt(
            sum((ta[i] - tb[i]) ** 2 for i in range(3))
        )
        translation_distances.append(distance)

        # Four quaternion components
        qa = a[3:7]
        qb = b[3:7]

        rotation_angles.append(
            quaternion_angle_degrees(qa, qb)
        )

    print("\nRaw translation difference:")
    print("Minimum:", min(translation_distances))
    print("Maximum:", max(translation_distances))
    print("Mean:", sum(translation_distances) / len(translation_distances))

    print("\nQuaternion angular difference:")
    valid_angles = [x for x in rotation_angles if math.isfinite(x)]

    if valid_angles:
        print("Minimum degrees:", min(valid_angles))
        print("Maximum degrees:", max(valid_angles))
        print("Mean degrees:", sum(valid_angles) / len(valid_angles))

    print(
        "\nNOTE: These are direct differences without coordinate-frame "
        "or scale alignment."
    )
    print(
        "They are diagnostic only until pose conventions and coordinate "
        "frames are confirmed."
    )


# ------------------------------------------------------------
# LOOP MATRIX AUDIT
# ------------------------------------------------------------

print("\n" + "=" * 75)
print("LOOP MATRIX AUDIT")
print("=" * 75)

matrix_path = FILES["loop_matrix"]
matrix = []
invalid_rows = []

if not os.path.isfile(matrix_path):
    print("Loop matrix file not found:", matrix_path)

else:
    with open(matrix_path, "r", encoding="utf-8-sig") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue

            try:
                row = [int(x.strip()) for x in line.strip().split(",")]

                if not row:
                    invalid_rows.append((line_no, "Empty row"))
                    continue

                if any(x not in (0, 1) for x in row):
                    invalid_rows.append((line_no, "Non-binary value"))
                    continue

                matrix.append(row)

            except Exception as exc:
                invalid_rows.append((line_no, str(exc)))

    print("Matrix file:", matrix_path)
    print("Invalid rows:", len(invalid_rows))

    if invalid_rows:
        print("Invalid row examples:", invalid_rows[:10])

    if matrix:
        row_lengths = [len(row) for row in matrix]
        nrows = len(matrix)
        consistent = len(set(row_lengths)) == 1
        ncols = row_lengths[0] if consistent else max(row_lengths)

        print("Matrix rows:", nrows)
        print("Minimum columns:", min(row_lengths))
        print("Maximum columns:", max(row_lengths))
        print("Consistent row lengths:", consistent)
        print("Square matrix:", consistent and nrows == ncols)

        values = Counter(x for row in matrix for x in row)
        print("Matrix value counts:", dict(sorted(values.items())))

        if consistent and nrows == ncols:
            diagonal_ones = sum(
                matrix[i][i] == 1 for i in range(nrows)
            )
            print("Diagonal ones:", diagonal_ones)

            asymmetric = sum(
                matrix[i][j] != matrix[j][i]
                for i in range(nrows)
                for j in range(i + 1, nrows)
            )
            print("Asymmetric off-diagonal pairs:", asymmetric)

            positive_per_row = [sum(row) for row in matrix]

            print("Positive entries per row - min:",
                  min(positive_per_row))
            print("Positive entries per row - max:",
                  max(positive_per_row))
            print("Positive entries per row - mean:",
                  sum(positive_per_row) / nrows)
            print("Total positive entries:", sum(positive_per_row))
            print("First 20 row sums:", positive_per_row[:20])

        else:
            print(
                "Square-matrix diagnostics skipped because the matrix "
                "is not consistently square."
            )

    else:
        print("No valid matrix data found.")


print("\n" + "=" * 75)
print("COLMAP REFERENCE AUDIT COMPLETE")
print("=" * 75)
