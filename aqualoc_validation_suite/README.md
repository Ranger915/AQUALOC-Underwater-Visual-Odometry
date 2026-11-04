# AQUALOC Sequence 06 — paper-validation suite

This package addresses the four outstanding submission checks without modifying prior benchmark data.

## Known measured baseline (from existing user-run evaluations)

- AQUALOC Archaeological Site Sequence 06: 3,385 grayscale frames.
- Original RAW + fixed CLAHE frontend: 60,774 pair records; adaptive v2: 30,387; calibrated / mutual evaluation from earlier work: 86,190 records.
- Adaptive v2 decision distribution: 808 RAW, 2,577 moderate CLAHE, 0 strong.
- Fixed control vs adaptive v2, harmonized success (`status==ok` and RANSAC inliers >=8): ORB 98.766% vs 98.766%; AKAZE 86.208% vs 86.050%; SIFT 99.447% vs 99.447%.
- Inlier-count changes adaptive minus fixed: ORB −47.96/pair (−14.3%), AKAZE −133.60 (−61.1%), SIFT −121.64 (−41.3%).
- Mixed RAW–CLAHE frame pairs: 52 per detector, and AKAZE had 16 fewer harmonized successes in these 52 pairs.
- Existing original vs fixed-control paired record checks: ORB/AKAZE exact match counts; SIFT 42 ratio and 40 inlier count mismatches across 10,129 pairs, 0 harmonized success mismatches.

## Step-by-step PowerShell (Windows)

Extract the ZIP folder `aqualoc_validation_suite` directly under
`C:\Users\Raaghav\Downloads\archaeo_sequence_6_raw_data\`.

```powershell
$root = 'C:\Users\Raaghav\Downloads\archaeo_sequence_6_raw_data'
$py = "$env:USERPROFILE\aqualoc-venv\Scripts\python.exe"
$suite = Join-Path $root 'aqualoc_validation_suite'
$out = Join-Path $root 'sequence6_paper_validation'
$fixed = Join-Path $root 'sequence6_fixed_clahe_control_full\adaptive_feature_pair_metrics.csv'
$adapt = Join-Path $root 'sequence6_adaptive_v2_full\adaptive_feature_pair_metrics.csv'
$original = Join-Path $root 'raw_data\sequence6_frontend_results\feature_pair_metrics.csv'
$images = Join-Path $root 'raw_data\images_sequence_6'
$package = Join-Path $root 'aqualoc_adaptive_package'

# 1. Paired moving-block bootstrap: all, stride 1, 5, 20; block sizes in frame anchors
& $py "$suite\01_adaptive_block_bootstrap.py" --fixed $fixed --adaptive $adapt --original $original --output $out --reps 1000 --blocks 10 50 100

# 2. Within-run preprocessing timing (reading excluded, 3 repeats / image)
& $py "$suite\02_preprocessing_timing.py" --image-dir $images --adaptive-package $package --output $out --repeats 3

# 3. Source / data / environment provenance manifest
& $py "$suite\03_reproducibility_audit.py" --root $root --adaptive-package $package --output $out

# 4. Locate the reference-geometry and mutual matching scripts / data
& $py "$suite\04_reference_pipeline_locator.py" --root $root --output $out

# 5. Print primary paired bootstrap results (block length 100, all strides)
Import-Csv "$out\adaptive_block_bootstrap.csv" | Where-Object {
    $_.block_frames -eq '100' -and $_.scope -eq 'all'
} | Select-Object detector,metric,paired_mean_difference,ci95_lower,ci95_upper,n_valid_pairs | Format-Table -AutoSize

# 6. Print timing summary
Import-Csv "$out\preprocessing_timing_summary.csv" | Format-Table -AutoSize
```

**Run order:** Step 1 statistical estimates; Step 2 timing; Step 3 provenance; Step 4 identify the remaining reference-scoring scripts.

## Interpretation and statistical safeguards

- The bootstrap is **paired** and resamples **contiguous blocks of first-frame anchors** rather than naively resampling individual pair records. All metrics from a sampled anchor are kept together. Block lengths 10, 50, and 100 allow sensitivity analysis. Report block 100 as a conservative sensitivity choice; it is not automatically a formally optimal block length.
- Success = `status == ok` AND `ransac_inliers >= 8` for both runs. Report percentage-point differences (100 × mean paired success difference).
- Inlier and ratio counts are averaged **over all paired records**, including failures. The inlier ratio includes zeros for failures. No imputation is performed.
- Epipolar residual comparisons are **restricted to pairs that succeed in both conditions**; they are an estimated fundamental-matrix proxy, not an independent COLMAP Sampson error. If failure sets differ, this conditional comparison may be selection biased.
- For adaptive decision transitions, the 52 mixed-type pairs can be dominated by a small number of adjacent segments; use descriptive per-segment analysis and avoid treating 52 as independent units.
- Bootstrap CIs are uncertainty under within-sequence resampling, not uncertainty across new AQUALOC sequences or diverse underwater scenes.
- Fixed control for feature equality: original and control SIFT slight mismatches need disclosure, not concealment.

## What is NOT yet possible without the calibrated/mutual scripts and match-level records

The 30,387-row frontend CSV contains counts and estimated-F residuals, **not** the coordinates and descriptors of each correspondence or independently scored reference labels. It cannot be converted to calibrated precision, recall, F1, AP, or reference Sampson error. To finish that part:

1. Run `04_reference_pipeline_locator.py`. Review `reference_pipeline_inventory.json` to identify the actual `*.py` scripts that generated `sequence6_calibrated_feature_results` and `sequence6_mutual_matching_results`, including their COLMAP pose loader and calibration-file sources.
2. Reuse the exact adaptive preprocessing `aqualoc_adaptive_package/adaptive_clahe.py` inside the **same** calibrated/mutual pipeline, adding `adaptive_v2` without altering detectors, matches, pair selection or reference geometry.
3. Reuse exactly the same labels, Sampson thresholds, class definitions and reporting scopes (all, local <=100, nonlocal >=200), with matching threshold grids of 8, 12 and 20 as used previously.
4. Export new per-pair metrics and detailed mutual correspondence records. Then calculate paired bootstrap confidence intervals over matched anchors and compare AP and F1 with fixed CLAHE.
5. Do not interpret the two COLMAP trajectory exports as independent ground truth: their 166 common positions are equivalent after Sim(3) (RMSE approx. 4e-12); one is a transformed representation of the other.
6. Do not generate ATE or SLAM-tracking conclusions without estimated camera pose trajectories from an actual VO/SLAM pipeline.

## Adaptive decision rule and provenance

The packaged v2 rule uses mean 8×8-grid local standard deviation threshold 10.5: local_std >= 10.5 → RAW; otherwise moderate CLAHE `clipLimit=2.0`, except strong CLAHE `clipLimit=3.0` if global_std <25 AND local_std <10. The actual run produced **zero** strong cases. Original fixed CLAHE uses `clipLimit=2.0`, `tileGridSize=8×8`. This v2 rule was inferred from prior observations and matches the **reported decision counts**; independently verify it against the original Claude-written method before final submission.

The threshold 10.5 was chosen from inspecting the same Sequence 06 histogram after v1's decision distribution; both v1 and v2 must be disclosed, and Sequence 06 should be treated as the threshold-development sequence, not an independent test dataset. Test another sequence before claiming portability.

## Ready-to-write limitations paragraph

"The adaptive threshold was selected after inspection of Sequence 06 image statistics and therefore constitutes sequence-specific, post-hoc configuration rather than independent parameter selection. Results cannot establish generalization across underwater scenes or AQUALOC sequences. Frame-pair observations share images and are temporally correlated; consequently, block-resampling sensitivity analyses and segment-level reporting are used instead of independent-pair significance assumptions. Frontend RANSAC inlier counts indicate consistency with an estimated fundamental matrix, not independently verified correspondence correctness or trajectory accuracy. The COLMAP reference exports are transform-equivalent and therefore cannot serve as two independent ground-truth estimates. Minor SIFT count differences were observed between otherwise equivalent fixed-CLAHE implementations."

## Extra control: temporal segments and exact mixed pairs

```powershell
& $py "$suite\05_transition_segments.py" --decisions "$root\sequence6_adaptive_v2_full\adaptive_frame_decisions.csv" --fixed $fixed --adaptive $adapt --output $out
Import-Csv "$out\preprocessing_segments.csv" | Format-Table -AutoSize
Import-Csv "$out\mixed_pair_details.csv" | Where-Object { $_.delta_success -eq '-1' } | Format-Table -AutoSize
```

This documents which transitions caused the AKAZE losses and counts mixed pairs by frame stride. Its output is descriptive; it does not treat transition pairs as independent replication.
