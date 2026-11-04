# AQUALOC V3: accuracy-first comparison (same engine, no new version)

## Purpose

Keep the established, stricter **minimum cheirality fraction 0.06** and compare `quality` against `descriptor_first` source selection. This isolates a selection-policy difference without relaxing geometric pose acceptance. It is an **exploratory, 600-frame pilot**, not guaranteed to improve error or yield 80% accepted translation updates.

## Required existing data

- `<root>/aqualoc_vo_v3_pilot/slam_engine_sequence6_v3.py` (dual-source version supporting `--source-policy`)
- `<root>/paper_research_draft/vo_pilot_dual_source/vo_sequence6_orb_clahe_v3.csv` (0.06 threshold, quality policy)
- `<root>/colmap_traj_sequence_6.txt`
- 600 Sequence 06 images in `<root>/raw_data/images_sequence_6`

## Run (PowerShell)

```powershell
$root = "$env:USERPROFILE\Downloads\archaeo_sequence_6_raw_data"
$py = "$env:USERPROFILE\aqualoc-venv\Scripts\python.exe"
$engine = "$root\aqualoc_vo_v3_pilot\slam_engine_sequence6_v3.py"
& $py $engine --root $root --max-frames 600 --algorithms ORB --preprocessings clahe `
  --max-features 1500 --flow-points 1800 --flow-fb-px 1.5 --ratio 0.75 `
  --ransac-px 1.5 --min-essential-inliers 15 --min-cheirality-inliers 10 `
  --min-cheirality-fraction 0.06 --min-parallax-deg 0.1 --max-anchor-gap 20 `
  --step-scale gap --source-policy descriptor_first `
  --out "$root\paper_research_draft\vo_pilot_descriptor_first"
if ($LASTEXITCODE -ne 0) { throw "V3 pilot failed" }
& $py "$root\aqualoc_v3_accuracy_experiment\compare_v3_selection.py" `
  --root $root --reference-convention twc --min-train 5 --min-test 3
if ($LASTEXITCODE -ne 0) { throw "Comparison failed" }
Import-Csv "$root\paper_research_draft\vo_selection_comparison\comparison_summary.csv" | Format-List
```

**IMPORTANT:** `twc` remains an unverified interpretation of the COLMAP reference format. Confirm source documentation before using these as publication results. Using `tcw` instead requires *re-running comparison under that independently established convention*, not cherry-picking the smaller error.

## Outputs

Under `paper_research_draft/vo_selection_comparison/`:

- `comparison_summary.csv` — paired-frame 60% calibration / 40% held-out Sim(3) comparison, acceptance and reference coverage.
- `heldout_errors.csv` and `heldout_error.png` — errors on identical frames.
- `<policy>_rejection_counts.csv`, `<policy>_candidate_diagnostics.csv`, `<policy>_source_by_acceptance.csv`.
- `<policy>_acceptance_timeline.png`.

No original file is modified. This compares alternative pose *selection*, not an upgraded scale estimator, and must not be called independent ground-truth ATE. The 600-frame sequence was used during previous model choices, so results are exploratory. Keep the 0.06 baseline even if one method performs worse. Follow with a longer frozen-parameter validation on unexamined frames or another sequence, not further holdout tuning.
