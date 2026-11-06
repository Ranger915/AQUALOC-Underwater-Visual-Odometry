import csv
import json
from pathlib import Path
from collections import defaultdict

ROOT = Path.home() / "Downloads" / "archaeo_sequence_6_raw_data"
INPUT = ROOT / "raw_data" / "sequence6_calibrated_reference" / "sequence6_calibrated_pair_manifest.csv"
OUT = INPUT.parent

GAP_BINS = [
    ("1 keyframe step", 1, 1),
    ("2 keyframe steps", 2, 2),
    ("3-5 keyframe steps", 3, 5),
    ("6-10 keyframe steps", 6, 10),
    ("11-20 keyframe steps", 11, 20),
    (">20 keyframe steps", 21, 10**9)
]

BASELINE_BINS = [
    ("<0.05", 0.0, 0.05),
    ("0.05-0.25", 0.05, 0.25),
    ("0.25-0.50", 0.25, 0.50),
    ("0.50-1.00", 0.50, 1.00),
    ("1.00-2.00", 1.00, 2.00),
    ("2.00-4.00", 2.00, 4.00),
    (">=4.00", 4.00, float("inf"))
]


def gap_category(frame_gap):
    steps = frame_gap // 20
    for name, low, high in GAP_BINS:
        if low <= steps <= high:
            return name
    return "unclassified"


def baseline_category(value):
    for name, low, high in BASELINE_BINS:
        if low <= value < high:
            return name
    return "unclassified"


def main():
    if not INPUT.is_file():
        raise FileNotFoundError(INPUT)

    records = []

    with INPUT.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)

        required = {
            "frame_id_i",
            "frame_id_j",
            "frame_gap",
            "colmap_overlap_label",
            "baseline_norm_export_units",
            "relative_rotation_degrees"
        }

        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Missing columns: {sorted(missing)}")

        for row in reader:
            gap = int(row["frame_gap"])
            baseline = float(row["baseline_norm_export_units"])
            overlap = int(row["colmap_overlap_label"])

            if gap <= 0 or gap % 20 != 0:
                raise ValueError(f"Unexpected frame gap: {gap}")

            if overlap not in (0, 1):
                raise ValueError(f"Invalid overlap label: {overlap}")

            row["gap_category"] = gap_category(gap)
            row["baseline_category"] = baseline_category(baseline)
            row["frame_gap"] = gap
            row["baseline"] = baseline
            row["overlap"] = overlap

            records.append(row)

    total = len(records)
    positives = sum(r["overlap"] for r in records)
    negatives = total - positives

    print("=" * 76)
    print("AQUALOC SEQUENCE 06 - PAIR DISTRIBUTION AUDIT")
    print("=" * 76)

    print("\n1. OVERALL DISTRIBUTION")
    print("-" * 76)
    print("Total pairs:", total)
    print("Overlap positive:", positives)
    print("Overlap negative:", negatives)
    print("Positive percentage:", round(100 * positives / total, 3))

    print("\n2. TEMPORAL SEPARATION")
    print("-" * 76)
    print(
        f"{'Frame gap category':<24}"
        f"{'Total':>10}{'Positive':>12}"
        f"{'Negative':>12}{'Positive %':>14}"
    )

    gap_summary = []

    for name, _, _ in GAP_BINS:
        subset = [r for r in records if r["gap_category"] == name]
        pos = sum(r["overlap"] for r in subset)
        neg = len(subset) - pos
        rate = 100 * pos / len(subset) if subset else 0.0

        print(
            f"{name:<24}{len(subset):>10}"
            f"{pos:>12}{neg:>12}{rate:>13.2f}%"
        )

        gap_summary.append({
            "gap_category": name,
            "total": len(subset),
            "positive": pos,
            "negative": neg,
            "positive_percent": rate
        })

    print("\n3. NON-LOCAL OVERLAP CANDIDATES")
    print("-" * 76)

    for threshold in (100, 200, 400, 600):
        subset = [r for r in records if r["frame_gap"] >= threshold]
        pos = sum(r["overlap"] for r in subset)

        print(
            f"Frame gap >= {threshold:>3}: "
            f"{len(subset):>6} total pairs, "
            f"{pos:>5} positive pairs"
        )

    print("\n4. BASELINE DISTRIBUTION")
    print("-" * 76)
    print(
        f"{'Baseline category':<20}"
        f"{'Total':>10}{'Positive':>12}"
        f"{'Negative':>12}{'Positive %':>14}"
    )

    baseline_summary = []

    for name, _, _ in BASELINE_BINS:
        subset = [r for r in records if r["baseline_category"] == name]
        pos = sum(r["overlap"] for r in subset)
        neg = len(subset) - pos
        rate = 100 * pos / len(subset) if subset else 0.0

        print(
            f"{name:<20}{len(subset):>10}"
            f"{pos:>12}{neg:>12}{rate:>13.2f}%"
        )

        baseline_summary.append({
            "baseline_category": name,
            "total": len(subset),
            "positive": pos,
            "negative": neg,
            "positive_percent": rate
        })

    print("\n5. SMALL-BASELINE PAIRS")
    print("-" * 76)

    for threshold in (0.01, 0.05, 0.10, 0.25):
        subset = [r for r in records if r["baseline"] < threshold]
        pos = sum(r["overlap"] for r in subset)

        print(
            f"Baseline < {threshold:.2f}: "
            f"{len(subset):>6} total, {pos:>5} positive"
        )

    print("\n6. LOCAL AND NON-LOCAL POSITIVE PAIRS")
    print("-" * 76)

    for name, condition in [
        ("Adjacent reconstructed frames", lambda g: g <= 20),
        ("Within 3 keyframe steps", lambda g: g <= 60),
        ("At least 10 keyframe steps apart", lambda g: g >= 200),
        ("At least 20 keyframe steps apart", lambda g: g >= 400)
    ]:
        count = sum(
            1 for r in records
            if r["overlap"] == 1 and condition(r["frame_gap"])
        )
        print(f"{name}: {count}")

    # Save auditable distribution tables.
    table_path = OUT / "sequence6_pair_distribution.csv"

    with table_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "dimension", "category", "total_pairs",
            "overlap_positive", "overlap_negative",
            "positive_percentage"
        ])

        for item in gap_summary:
            writer.writerow([
                "temporal_gap",
                item["gap_category"],
                item["total"],
                item["positive"],
                item["negative"],
                f'{item["positive_percent"]:.6f}'
            ])

        for item in baseline_summary:
            writer.writerow([
                "baseline",
                item["baseline_category"],
                item["total"],
                item["positive"],
                item["negative"],
                f'{item["positive_percent"]:.6f}'
            ])

    summary = {
        "total_pairs": total,
        "positive_pairs": positives,
        "negative_pairs": negatives,
        "positive_percentage": 100 * positives / total,
        "temporal_gap_distribution": gap_summary,
        "baseline_distribution": baseline_summary
    }

    json_path = OUT / "sequence6_pair_distribution.json"
    json_path.write_text(json.dumps(summary, indent=4), encoding="utf-8")

    print("\n7. AUDIT OUTPUTS")
    print("-" * 76)
    print(table_path)
    print(json_path)
    print("\nOriginal calibrated manifest modified: NO")
    print("Original feature benchmark modified: NO")
    print("=" * 76)


if __name__ == "__main__":
    main()
