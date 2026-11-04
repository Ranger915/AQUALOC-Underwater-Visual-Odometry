# AQUALOC Sequence 06: Adaptive CLAHE v2 reference-geometry and mutual validation

This package is a minimal modification of the supplied calibrated-feature and mutual-matching scripts. It runs **only adaptive_v2** and keeps the original reference manifest, detectors, matching algorithms, thresholds, and raw/CLAHE result directories untouched.

## Files
- `adaptive_clahe.py`: original reusable v1/v2 function, byte-for-byte copied from the supplied ZIP.
- `aqualoc_sequence6_calibrated_feature_adaptive_v2.py`: original reference-geometry evaluator, with adaptive_v2 preprocessing and isolated output.
- `aqualoc_sequence6_mutual_matching_adaptive_v2.py`: original mutual evaluator, with adaptive_v2 preprocessing and isolated output.

## Paths assumed by the supplied scripts
- Project: `C:\Users\Raaghav\Downloads\archaeo_sequence_6_raw_data`
- Images: `<project>\raw_data\images_sequence_6`
- Reference manifest: `<project>\raw_data\sequence6_calibrated_reference\sequence6_calibrated_pair_manifest.csv`
- Calibrated adaptive output: `<project>\sequence6_adaptive_v2_calibrated_reference_results`
- Mutual adaptive output: `<project>\sequence6_adaptive_v2_mutual_matching_results`

The scripts use `Path.home()` so this works under the user's home directory. Copy the three Python files together into one folder inside the project (e.g. `aqualoc_adaptive_reference_v2_ready`).

## PowerShell
```powershell
$root = "$env:USERPROFILE\Downloads\archaeo_sequence_6_raw_data"
$py = "$env:USERPROFILE\aqualoc-venv\Scripts\python.exe"
$dir = Join-Path $root "aqualoc_adaptive_reference_v2_ready"

Test-Path (Join-Path $root "raw_data\sequence6_calibrated_reference\sequence6_calibrated_pair_manifest.csv")
Test-Path (Join-Path $root "raw_data\images_sequence_6\frame000000.png")

& $py -m py_compile (Join-Path $dir "adaptive_clahe.py") (Join-Path $dir "aqualoc_sequence6_calibrated_feature_adaptive_v2.py") (Join-Path $dir "aqualoc_sequence6_mutual_matching_adaptive_v2.py")

& $py (Join-Path $dir "aqualoc_sequence6_calibrated_feature_adaptive_v2.py")
& $py (Join-Path $dir "aqualoc_sequence6_mutual_matching_adaptive_v2.py")
```

The scripts do not overwrite completed adaptive result CSVs. They will refuse to run if their main adaptive output CSV already exists. Preserve/rename a completed results folder before re-running.

## Checks after both runs
```powershell
$cal = Import-Csv "$root\sequence6_adaptive_v2_calibrated_reference_results\sequence6_reference_feature_metrics.csv"
$mut = Import-Csv "$root\sequence6_adaptive_v2_mutual_matching_results\sequence6_mutual_feature_metrics.csv"
"Calibrated rows: $($cal.Count) (expected 43095)"
"Mutual rows: $($mut.Count) (expected 43095)"
$cal | Group-Object detector,preprocessing | Select-Object Name,Count
$mut | Group-Object detector,preprocessing | Select-Object Name,Count

$cal | Where-Object { $_.preprocessing -ne 'adaptive_v2' } | Measure-Object
$mut | Where-Object { $_.preprocessing -ne 'adaptive_v2' } | Measure-Object
```

Expected per detector **14,365 pairs**, or **43,095** rows for calibrated and another **43,095** for mutual. The reference manifest includes 170 keyframes, not all 3,385 original video frames.

## Experimental comparability notes
- Adaptive-v2 decision is unchanged: RAW iff mean 8x8 local grid intensity standard deviation >= 10.5; otherwise strong (global<25 and local<10, clip=3) or moderate (clip=2) CLAHE.
- The source adaptive rule is **sequence-specific and chosen after seeing v1's decision distribution**. It should not be described as prospective/generalized validation.
- Crucially, **808 RAW / 2,577 CLAHE is the distribution over 3,385 frames**, not necessarily over the 170 COLMAP keyframes. Do not assume the same percentages.
- Reference scoring uses existing intrinsic/distortion numbers baked into the original scripts and COLMAP-derived fundamental matrices. Verify the associated official calibration file before stating independent camera-calibration provenance.
- The benchmark provides overlap-detection precision/recall/F1/AP using **COLMAP-derived overlap labels**. They are reference-derived, not independent physical ground truth; SIFT-reference bias remains possible.
- No ATE or full SLAM pose error is calculated by these scripts.
- Results from this package are not yet evaluated on the user's machine; successful syntax/import and synthetic preprocessing tests are not numerical matching validation.
