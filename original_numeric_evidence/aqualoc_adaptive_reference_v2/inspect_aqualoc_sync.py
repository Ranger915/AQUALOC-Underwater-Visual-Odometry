import os
import csv
import bisect
import statistics
from collections import Counter

ROOT = os.path.join(
    os.path.expanduser("~"),
    "Downloads",
    "archaeo_sequence_6_raw_data",
    "raw_data"
)

FILES = {
    "Camera": "img_sequence_6.csv",
    "Depth": "depth_sequence_6.csv",
    "IMU": "imu_sequence_6.csv",
    "Magnetometer": "mag_sequence_6.csv"
}


def load_timestamps(filename):
    path = os.path.join(ROOT, filename)
    timestamps = []

    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        next(reader, None)

        for row in reader:
            if row:
                try:
                    timestamps.append(int(row[0].strip()))
                except (ValueError, IndexError):
                    pass

    return timestamps


def ms(value):
    return value / 1e6


def describe_intervals(name, timestamps):
    intervals = [
        timestamps[i + 1] - timestamps[i]
        for i in range(len(timestamps) - 1)
    ]

    intervals_ms = [ms(x) for x in intervals]

    print("\n" + "-" * 70)
    print(name)
    print("-" * 70)
    print("Samples:", len(timestamps))
    print("Duration [s]:",
          (timestamps[-1] - timestamps[0]) / 1e9)
    print("Mean interval [ms]:", statistics.mean(intervals_ms))
    print("Median interval [ms]:", statistics.median(intervals_ms))
    print("5th percentile [ms]:",
          sorted(intervals_ms)[int(0.05 * (len(intervals_ms) - 1))])
    print("95th percentile [ms]:",
          sorted(intervals_ms)[int(0.95 * (len(intervals_ms) - 1))])
    print("Minimum interval [ms]:", min(intervals_ms))
    print("Maximum interval [ms]:", max(intervals_ms))
    print("Mean-rate estimate [Hz]:",
          1e9 / statistics.mean(intervals))
    print("Median-rate estimate [Hz]:",
          1e9 / statistics.median(intervals))

    rounded = Counter(round(x, 2) for x in intervals_ms)
    print("Most common intervals [ms]:", rounded.most_common(10))


def nearest_differences(camera_ts, sensor_ts):
    differences = []

    for t in camera_ts:
        index = bisect.bisect_left(sensor_ts, t)
        candidates = []

        if index < len(sensor_ts):
            candidates.append(abs(sensor_ts[index] - t))

        if index > 0:
            candidates.append(abs(sensor_ts[index - 1] - t))

        if candidates:
            differences.append(min(candidates))

    return [ms(x) for x in differences]


print("=" * 76)
print("AQUALOC SEQUENCE 06 - SENSOR TIMESTAMP SYNCHRONIZATION AUDIT")
print("=" * 76)

streams = {
    name: load_timestamps(filename)
    for name, filename in FILES.items()
}

for name, timestamps in streams.items():
    if len(timestamps) < 2:
        raise RuntimeError(f"Insufficient timestamps in {name}")

    print(
        f"\n{name}: monotonically increasing = "
        f"{all(timestamps[i+1] > timestamps[i] for i in range(len(timestamps)-1))}"
    )
    print("Duplicate timestamps:",
          len(timestamps) - len(set(timestamps)))

    describe_intervals(name, timestamps)


camera = streams["Camera"]

print("\n" + "=" * 76)
print("CAMERA-TO-SENSOR NEAREST TIMESTAMP DIFFERENCES")
print("=" * 76)

for name in ["Depth", "IMU", "Magnetometer"]:
    sensor = streams[name]
    differences = nearest_differences(camera, sensor)

    print("\nCamera versus", name)
    print("Camera frames:", len(camera))
    print("Sensor samples:", len(sensor))
    print("Sensor starts [ms] relative to camera:",
          ms(sensor[0] - camera[0]))
    print("Sensor ends [ms] relative to camera:",
          ms(sensor[-1] - camera[-1]))
    print("Nearest difference median [ms]:",
          statistics.median(differences))
    print("Nearest difference 95th percentile [ms]:",
          sorted(differences)[int(0.95 * (len(differences)-1))])
    print("Nearest difference maximum [ms]:", max(differences))

    for threshold in [2.5, 5, 10, 20]:
        count = sum(x <= threshold for x in differences)
        print(
            f"Camera frames with nearest sample <= {threshold} ms: "
            f"{count}/{len(camera)} "
            f"({100*count/len(camera):.2f}%)"
        )

    covered = sum(sensor[0] <= t <= sensor[-1] for t in camera)
    print("Camera timestamps inside sensor time range:",
          f"{covered}/{len(camera)}")


print("\n" + "=" * 76)
print("SYNCHRONIZATION AUDIT COMPLETE")
print("=" * 76)
