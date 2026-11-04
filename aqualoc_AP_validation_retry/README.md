# AQUALOC Sequence 06 — final mutual AP statistical validation

## What is covered

The previous work already established the paired F1 bootstrap, frontend RANSAC/inlier bootstrap, fixed-control equivalence, preprocessing timing, reproducibility manifest and mixed-pair transition analysis. **This package completes the outstanding paired bootstrap for mutual-matching average precision (AP)**: both absolute mutual reference-inlier count and normalized mutual reference-inlier fraction.

AP is calculated from individual pair-level scores, **not from the summary CSV**. Existing summary APs are used only to verify the computation reproduces the original scoring definition. If it doesn't, the script fails rather than output misleading confidence intervals.

## PowerShell steps

Unzip `aqualoc_final_ap_validation.zip` into:

`C:\Users\Raaghav\Downloads\archaeo_sequence_6_raw_data`

Run:

```powershell
$root = "$env:USERPROFILE\Downloads\archaeo_sequence_6_raw_data"
$py = "$env:USERPROFILE\aqualoc-venv\Scripts\python.exe"
$script = Join-Path $root "aqualoc_final_ap_validation\paired_ap_bootstrap.py"

Test-Path $script
Test-Path "$root\raw_data\sequence6_mutual_matching_results\sequence6_mutual_feature_metrics.csv"
Test-Path "$root\sequence6_adaptive_v2_mutual_matching_results\sequence6_mutual_feature_metrics.csv"

& $py $script --root $root --reps 1000 --blocks 10
if ($LASTEXITCODE -ne 0) { throw "AP validation failed; inspect console output before making claims" }

Import-Csv "$root\sequence6_paper_validation\adaptive_v2_mutual_ap_paired_bootstrap.csv" |
  Where-Object { $_.block_keyframes -eq '10' -and $_.metric -eq 'average_precision_count' } |
  Select-Object detector,scope,fixed_ap,adaptive_v2_ap,paired_difference_adaptive_minus_fixed,ci95_lower,ci95_upper |
  Format-Table -AutoSize
```

Optional sensitivity check (more work):

```powershell
& $py $script --root $root --reps 1000 --blocks 1 5 10
```

You can view normalized-fraction AP in the same output file by filtering `metric -eq 'average_precision_fraction'`.

## Outputs

All generated in `sequence6_paper_validation`:

- `adaptive_v2_mutual_ap_paired_bootstrap.csv`: point estimates, paired AP differences, 95% percentile confidence intervals and bootstrap frequencies.
- `adaptive_v2_ap_definition_validation.csv`: computed AP values versus saved benchmark summary values, with absolute errors for each detector/scope/metric/method.
- `adaptive_v2_mutual_ap_bootstrap_metadata.json`: methods, sampling limitations, parameters, input locations and provenance.

## Statistical method

- Fixed and adaptive use the same detector, ordered keyframe pair and COLMAP-derived binary overlap label.
- AP is non-interpolated area under the stepwise precision–recall curve, equivalent to scikit-learn's `average_precision_score` with equal sample weights and tie handling.
- Per-pair rankings are fixed; a paired circular moving-block bootstrap samples *first-frame keyframe anchors* in blocks of 10. Both methods use the same replicate weights.
- Report `adaptive_v2 - fixed CLAHE`. A 95% interval excluding zero gives evidence of a difference **under this resampling scheme**, not proof of true cross-sequence generalization.
- This block-bootstrap design does **not** capture all dependencies arising from repeated second-frame use. Interpret confidence intervals as sensitivity estimates.
- The adaptive v2 threshold was selected after inspecting Sequence 06's local-contrast distribution; do not characterize this as external prospective validation.

## Important integrity controls

The program refuses to report results unless:
1. Exactly 43,095 pair records align between fixed and adaptive across ORB, AKAZE-MLDB and SIFT.
2. All reference overlap labels match.
3. Recalculated count AP and fraction AP match both saved summary files, per scope/detector, to `1e-8` (by default).
4. Each analysis scope contains positive and negative reference pairs.

If the AP definition check fails, do **not** loosen the tolerance to mask the mismatch. Instead inspect the original evaluator's AP definition and scoring/filtering first.

## Final paper scope

This is a **feature correspondence** study, not a completed SLAM pose-estimation benchmark. Do not claim improved trajectory ATE, reconstruction accuracy, navigation outcomes or cross-dataset generalization. Report adaptive v1 and v2 as exploratory sequence-specific decisions, and retain raw/fixed CLAHE as baselines. The calibrated and mutual labels remain COLMAP-derived references, not independent manually verified scene ground truth.
