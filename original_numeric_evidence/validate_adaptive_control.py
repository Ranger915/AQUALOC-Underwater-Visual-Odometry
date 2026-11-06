import csv
from collections import defaultdict, Counter
from pathlib import Path

ROOT = Path(r"C:\Users\Raaghav\Downloads\archaeo_sequence_6_raw_data")

paths = {
    "original": ROOT / "raw_data/sequence6_frontend_results/feature_pair_metrics.csv",
    "fixed_control": ROOT / "sequence6_fixed_clahe_control_full/adaptive_feature_pair_metrics.csv",
    "adaptive_v2": ROOT / "sequence6_adaptive_v2_full/adaptive_feature_pair_metrics.csv",
}
decisions_path = ROOT / "sequence6_adaptive_v2_full/adaptive_frame_decisions.csv"

out = ROOT / "sequence6_adaptive_v2_comparison"
out.mkdir(exist_ok=True)

def load(path, original=False):
    records = {}
    with path.open(newline="", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            if original and r["preprocessing"].lower() != "clahe":
                continue
            det = r["detector"].upper().replace("AKAZE_MLDB", "AKAZE")
            a = int(r["frame_a_id"] if original else r["frame_id_1"])
            b = int(r["frame_b_id"] if original else r["frame_id_2"])
            stride = int(r["pair_stride"] if original else r["frame_stride"])
            key = (det, a, b, stride)
            if key in records:
                raise ValueError(f"Duplicate record: {key}")
            records[key] = r
    return records

records = {
    name: load(path, name == "original")
    for name, path in paths.items()
}

with decisions_path.open(newline="", encoding="utf-8-sig") as f:
    decisions = {
        int(r["frame_id"]): r["decision"]
        for r in csv.DictReader(f)
    }

# Harmonized definition across all runs:
# F estimated successfully AND at least 8 RANSAC inliers.
def success(r):
    return (
        r["status"] == "ok"
        and int(float(r["ransac_inliers"])) >= 8
    )

def pair_type(a, b):
    modes = (decisions[a], decisions[b])
    if modes == ("raw", "raw"):
        return "RAW_RAW"
    if all(x == "moderate_clahe" for x in modes):
        return "CLAHE_CLAHE"
    return "MIXED_OR_OTHER"

common = set.intersection(*(set(x) for x in records.values()))
print("Common pairs across all three datasets:", len(common))
print("Original:", len(records["original"]))
print("Control:", len(records["fixed_control"]))
print("Adaptive:", len(records["adaptive_v2"]))

if len(common) != 30387:
    raise RuntimeError("Incomplete pair alignment; stop and inspect")

print("\nHARMONIZED SUCCESS: >=8 RANSAC INLIERS")
print("-" * 75)

for det in ("ORB", "AKAZE", "SIFT"):
    keys = [k for k in common if k[0] == det]
    print("\n", det)
    for name in ("original", "fixed_control", "adaptive_v2"):
        n = sum(success(records[name][k]) for k in keys)
        print(f"  {name:15} {n:5}/{len(keys)} = {100*n/len(keys):.3f}%")

print("\nORIGINAL VS FIXED CONTROL: PAIR-LEVEL CONSISTENCY")
print("-" * 75)
for det in ("ORB", "AKAZE", "SIFT"):
    keys = [k for k in common if k[0] == det]
    inlier_mismatches = sum(
        int(float(records["original"][k]["ransac_inliers"]))
        != int(float(records["fixed_control"][k]["ransac_inliers"]))
        for k in keys
    )
    match_mismatches = sum(
        int(float(records["original"][k]["ratio_matches"]))
        != int(float(records["fixed_control"][k]["ratio_matches"]))
        for k in keys
    )
    success_mismatches = sum(
        success(records["original"][k])
        != success(records["fixed_control"][k])
        for k in keys
    )
    print(
        f"{det:6} Inlier mismatches={inlier_mismatches}, "
        f"ratio-match mismatches={match_mismatches}, "
        f"harmonized-status mismatches={success_mismatches}"
    )

rows = []
for det in ("ORB", "AKAZE", "SIFT"):
    for stride in (1, 5, 20, "all"):
        for kind in ("RAW_RAW", "CLAHE_CLAHE", "MIXED_OR_OTHER", "all"):
            keys = [
                k for k in common
                if k[0] == det
                and (stride == "all" or k[3] == stride)
                and (kind == "all" or pair_type(k[1], k[2]) == kind)
            ]
            if not keys:
                continue

            fixed = [records["fixed_control"][k] for k in keys]
            adaptive = [records["adaptive_v2"][k] for k in keys]

            fs = sum(success(x) for x in fixed)
            ads = sum(success(x) for x in adaptive)

            fi = sum(float(x["ransac_inliers"]) for x in fixed) / len(keys)
            ai = sum(float(x["ransac_inliers"]) for x in adaptive) / len(keys)

            rows.append({
                "detector": det,
                "stride": stride,
                "pair_type": kind,
                "pairs": len(keys),
                "fixed_success_pct": round(100 * fs / len(keys), 4),
                "adaptive_success_pct": round(100 * ads / len(keys), 4),
                "success_delta_pp": round(100 * (ads - fs) / len(keys), 4),
                "fixed_mean_inliers": round(fi, 4),
                "adaptive_mean_inliers": round(ai, 4),
                "inlier_delta": round(ai - fi, 4),
            })

csv_path = out / "harmonized_stratified_validation.csv"
with csv_path.open("w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=rows[0].keys())
    w.writeheader()
    w.writerows(rows)

print("\nADAPTIVE VS FIXED CONTROL: BY PAIR TYPE")
print("-" * 75)
for r in rows:
    if r["stride"] == "all" and r["pair_type"] != "all":
        print(
            f'{r["detector"]:6} {r["pair_type"]:15} '
            f'n={r["pairs"]:5} '
            f'SuccessDelta={r["success_delta_pp"]:+8.3f} pp '
            f'InlierDelta={r["inlier_delta"]:+9.2f}'
        )

print("\nSaved:", csv_path)
