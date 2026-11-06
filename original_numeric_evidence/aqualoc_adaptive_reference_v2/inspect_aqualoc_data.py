import os
import csv
import math
import statistics
from collections import Counter

ROOT = os.path.join(
    os.path.expanduser("~"),
    "Downloads",
    "archaeo_sequence_6_raw_data"
)

RAW = os.path.join(ROOT, "raw_data")
IMAGE_DIR = os.path.join(RAW, "images_sequence_6")

FILES = [
    "img_sequence_6.csv",
    "depth_sequence_6.csv",
    "imu_sequence_6.csv",
    "mag_sequence_6.csv"
]


def inspect_csv(filename):
    path = os.path.join(RAW, filename)

    print("\n" + "=" * 76)
    print("FILE:", filename)
    print("=" * 76)

    if not os.path.isfile(path):
        print("FILE NOT FOUND")
        return

    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        rows = [
            [x.strip() for x in row]
            for row in reader
            if row and any(x.strip() for x in row)
        ]

    print("Header:", header)
    print("Data rows:", len(rows))
    print("First 3 records:")
    for row in rows[:3]:
        print(" ", row)

    if not rows:
        return

    timestamps = []
    invalid_timestamps = 0

    for row in rows:
        try:
            timestamps.append(int(row[0]))
        except (ValueError, IndexError):
            invalid_timestamps += 1

    print("Valid timestamps:", len(timestamps))
    print("Invalid timestamps:", invalid_timestamps)

    if timestamps:
        intervals = [
            timestamps[i + 1] - timestamps[i]
            for i in range(len(timestamps) - 1)
        ]

        print("First timestamp [ns]:", timestamps[0])
        print("Last timestamp [ns]:", timestamps[-1])
        print("Timestamp range [s]:",
              (timestamps[-1] - timestamps[0]) / 1e9)
        print("Monotonically increasing:",
              all(interval > 0 for interval in intervals))
        print("Duplicate timestamps:",
              len(timestamps) - len(set(timestamps)))

        if intervals:
            print("Interval min [ms]:", min(intervals) / 1e6)
            print("Interval median [ms]:",
                  statistics.median(intervals) / 1e6)
            print("Interval max [ms]:", max(intervals) / 1e6)

    if filename == "img_sequence_6.csv":
        frame_ids = []
        for row in rows:
            if len(row) >= 2:
                frame_ids.append(row[1])

        print("Unique image filenames:", len(set(frame_ids)))
        print("First image:", frame_ids[0] if frame_ids else None)
        print("Last image:", frame_ids[-1] if frame_ids else None)

    elif filename == "depth_sequence_6.csv":
        values = []
        for row in rows:
            try:
                values.append(float(row[1]))
            except (ValueError, IndexError):
                pass

        if values:
            print("Valid depth values:", len(values))
            print("Depth min [m]:", min(values))
            print("Depth max [m]:", max(values))
            print("Depth mean [m]:", statistics.mean(values))
            print("Depth median [m]:", statistics.median(values))
            print("Non-finite depth values:",
                  sum(not math.isfinite(x) for x in values))

    elif filename in ("imu_sequence_6.csv", "mag_sequence_6.csv"):
        numeric_columns = list(zip(*[
            [float(x) for x in row[1:]]
            for row in rows
            if len(row) == len(header)
        ]))

        for name, column in zip(header[1:], numeric_columns):
            print(
                name,
                "| min:", min(column),
                "| max:", max(column),
                "| mean:", statistics.mean(column)
            )


print("=" * 76)
print("AQUALOC SEQUENCE 06 - SENSOR DATA INSPECTION")
print("=" * 76)

for filename in FILES:
    inspect_csv(filename)


# ------------------------------------------------------------
# IMAGE DIMENSIONS
# ------------------------------------------------------------

print("\n" + "=" * 76)
print("IMAGE DIMENSION INSPECTION")
print("=" * 76)

try:
    import cv2

    print("OpenCV version:", cv2.__version__)

    for i in [0, 1, 20, 100, 3384]:
        filename = f"frame{i:06d}.png"
        path = os.path.join(IMAGE_DIR, filename)

        image = cv2.imread(path, cv2.IMREAD_UNCHANGED)

        if image is None:
            print(filename, ": NOT FOUND OR UNREADABLE")
        else:
            print(
                filename,
                "| width:", image.shape[1],
                "| height:", image.shape[0],
                "| shape:", image.shape,
                "| dtype:", image.dtype
            )

except ImportError:
    print("OpenCV is not available in this Python environment.")


# ------------------------------------------------------------
# CALIBRATION FILE
# ------------------------------------------------------------

print("\n" + "=" * 76)
print("DOWNLOADED CALIBRATION CANDIDATE")
print("=" * 76)

yaml_path = os.path.join(ROOT, "Aqualoc_ORB_SLAM3_reference.yaml")

if os.path.isfile(yaml_path):
    with open(yaml_path, "r", encoding="utf-8") as f:
        for line in list(f)[:35]:
            print(line.rstrip())
else:
    print("Candidate YAML not found:", yaml_path)

print("\nINSPECTION COMPLETE")
