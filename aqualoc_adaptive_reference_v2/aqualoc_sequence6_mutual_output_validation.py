import csv
import gzip
from pathlib import Path
from collections import defaultdict

ROOT = Path.home() / "Downloads" / "archaeo_sequence_6_raw_data"
OUT = ROOT / "raw_data" / "sequence6_mutual_matching_results"

print("=" * 76)
print("MUTUAL MATCHING OUTPUT VALIDATION")
print("=" * 76)

files = sorted(OUT.glob("mutual_matches_*.csv.gz"))

if len(files) != 6:
    raise RuntimeError(f"Expected 6 correspondence files, found {len(files)}")

for path in files:
    with gzip.open(path, "rt", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        count = 0

        for row in reader:
            count += 1

            if int(row["query_keypoint_index"]) < 0:
                raise ValueError("Invalid query keypoint index")

            if int(row["train_keypoint_index"]) < 0:
                raise ValueError("Invalid train keypoint index")

    print(f"{path.name}: {count} mutual matches, {path.stat().st_size} bytes")

metrics_path = OUT / "sequence6_mutual_feature_metrics.csv"

aggregate = defaultdict(lambda: {
    "rows": 0,
    "pairs_with_matches": 0,
    "mutual_matches": 0,
    "positive_pairs": 0,
    "positive_pairs_with_matches": 0
})

with metrics_path.open("r", newline="", encoding="utf-8-sig") as f:
    for row in csv.DictReader(f):
        key = (row["detector"], row["preprocessing"])
        item = aggregate[key]

        count = int(row["mutual_matches"])
        label = int(row["overlap_label"])

        item["rows"] += 1
        item["mutual_matches"] += count

        if count:
            item["pairs_with_matches"] += 1

        if label:
            item["positive_pairs"] += 1
            if count:
                item["positive_pairs_with_matches"] += 1

print("\nCONFIGURATION SUMMARY")
print("-" * 76)

for key, item in sorted(aggregate.items()):
    print(
        f"{key[0]:<13} {key[1]:<8} "
        f"rows={item['rows']:>5} "
        f"pairs_with_matches={item['pairs_with_matches']:>5} "
        f"total_matches={item['mutual_matches']:>8} "
        f"positive_pairs_with_matches="
        f"{item['positive_pairs_with_matches']:>5}/"
        f"{item['positive_pairs']}"
    )

if any(item["mutual_matches"] == 0 for item in aggregate.values()):
    raise RuntimeError(
        "At least one configuration has zero mutual matches. "
        "Inspect the matching outputs before using the experiment."
    )

print("\nVALIDATION PASSED")
print("=" * 76)
