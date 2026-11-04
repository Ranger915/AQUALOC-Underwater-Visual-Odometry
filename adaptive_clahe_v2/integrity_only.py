"""Run only the integrity check on saved adaptive results (no benchmark rerun)."""
import argparse
import csv
from pathlib import Path

import aqualoc_sequence6_frontend as f

p = argparse.ArgumentParser()
p.add_argument("--image-dir", required=True)
p.add_argument("--results-dir", required=True, help="folder holding adaptive_pair_metrics.csv")
p.add_argument("--baseline-csv", required=True)
p.add_argument("--pair-strides", nargs="+", type=int, default=[1, 5, 20])
a = p.parse_args()

out = Path(a.results_dir)
if not Path(a.baseline_csv).is_file():
    p.error(f"--baseline-csv not found: {a.baseline_csv}")
paths = sorted([q for q in Path(a.image_dir).glob("*.png") if f.frame_id(q) >= 0], key=f.frame_id)
with (out / "adaptive_pair_metrics.csv").open(newline="", encoding="utf-8") as fh:
    rows = list(csv.DictReader(fh))
with (out / "frame_decisions.csv").open(newline="", encoding="utf-8") as fh:
    decisions = {r["frame"]: r for r in csv.DictReader(fh)}
f.integrity_report(rows, paths, sorted(set(a.pair_strides)), decisions, a.baseline_csv, out)
