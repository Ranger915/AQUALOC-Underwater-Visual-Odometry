# Adaptive CLAHE preprocessing for the AQUALOC Sequence 06 feature benchmark

Status: draft. Image-statistics analysis and decision distributions are done for v1 and v2. The matching benchmark for v2 has not been run or analysed yet, so this document contains no matching results and makes no claim that adaptive preprocessing improves anything.

## 1. Objective

Add a contrast-driven adaptive preprocessing condition to the existing front-end feature benchmark on AQUALOC Sequence 06. The aim is to enhance low-contrast underwater frames while leaving frames that already carry enough contrast untouched. Adaptive CLAHE is an additional experimental condition. It does not replace the existing RAW and fixed-CLAHE conditions, and baseline outputs are never overwritten.

## 2. Fixed benchmark configuration

These values come from the original `aqualoc_sequence6_frontend.py` and were not changed.

| Item | Value |
|---|---|
| Dataset | AQUALOC Sequence 06 |
| Images | 3,385 grayscale PNG frames, 968 x 608 |
| Detectors | ORB, SIFT, AKAZE (MLDB descriptor) |
| Max features | 1,500 (keypoints beyond the limit are cut by detector response) |
| Matcher | Brute force, k-NN with k = 2; Hamming norm for ORB and AKAZE, L2 for SIFT |
| Lowe ratio threshold | 0.75 |
| Geometry | `cv2.findFundamentalMat`, FM_RANSAC, threshold 1.5 px, confidence 0.99 |
| Minimum matches for geometry | 8 ratio matches; 8 inliers for the error and coverage metrics |
| Frame strides | 1, 5, 20 |
| Baseline preprocessing | RAW, and fixed CLAHE with clip limit 2.0 and an 8 x 8 tile grid |

## 3. Metrics

### 3.1 Computed by the current script (per image pair)

| Column | Meaning |
|---|---|
| `keypoints_a`, `keypoints_b` | Keypoints detected in each frame |
| `raw_knn_matches` | Descriptor pairs that returned two nearest neighbours |
| `ratio_matches` | Matches passing the 0.75 ratio test |
| `ransac_inliers` | Inliers of the RANSAC fundamental matrix |
| `inlier_ratio` | `ransac_inliers / ratio_matches` |
| `mean_symmetric_epipolar_error_px` | Mean of the two point-to-epipolar-line distances over inliers. This is a point-to-line residual proxy from an uncalibrated fundamental matrix. It is not the Sampson error. |
| `grid_cells_covered_4x4` | Number of 4 x 4 image cells containing at least one inlier |
| `extraction_time_pair_s` | Detection plus description time for both frames |
| `matching_time_s`, `geometry_time_s` | Matching time and RANSAC time |
| `status` | `ok`, `insufficient_descriptors`, `fewer_than_8_ratio_matches`, `few_ransac_inliers`, `fundamental_matrix_failed` |

Pairs are formed per detector and per preprocessing mode for every stride in {1, 5, 20}. Preprocessing time is not included in `extraction_time_pair_s`, in the baseline or in the adaptive runs.

### 3.2 Specified for the full evaluation but not in this script

The evaluation plan lists mutual matches, precision, recall and F1, average precision, Sampson error, local versus non-local matching performance, and the existing statistical validation. These belong to the calibrated reference-geometry and mutual-matching scripts, which were not modified. Those scripts can import the same preprocessing from `adaptive_clahe.py` so all three conditions are preprocessed identically.

### 3.3 Image statistics used by the adaptive method

For every frame, from the raw grayscale image only:

1. Global intensity standard deviation.
2. Mean local standard deviation over an 8 x 8 grid (cells of 121 x 76 px).

Per-frame decision records contain the frame ID, both statistics, the selected mode, the clip limit, the tile grid, and the time taken to compute the statistics.

## 4. Experimental controls

- Raw and fixed-CLAHE code paths are untouched. A default run behaves as before.
- Adaptive decisions use only the current image's intensity statistics. No COLMAP data, ground truth, matching results or RANSAC results enter the decision.
- The decision is computed once per frame and shared by all three detectors.
- Frame selection, strides, detector settings, matching thresholds, RANSAC settings and metrics are identical across conditions.
- Adaptive runs write to a separate folder, and the script refuses to run in a folder that contains the baseline `feature_pair_metrics.csv`.
- Thresholds are fixed constants in `adaptive_clahe.py`. No threshold was tuned against F1, inlier counts or any other evaluation outcome.

## 5. Adaptive v1 (global plus local gate)

### 5.1 Rule

Fixed thresholds: global 35 and 25, local 15 and 10, CLAHE tile grid 8 x 8.

1. Global std >= 35 and local std >= 15: keep RAW.
2. Global std < 25 and local std < 10: strong CLAHE, clip limit 3.0.
3. Otherwise: moderate CLAHE, clip limit 2.0.

The specification's fourth rule ("otherwise RAW") can never apply. After rule 1 fails, at least one statistic is below its normal threshold, so rule 3 always catches the frame.

### 5.2 Decision distribution (3,385 frames)

| Mode | Frames | Share |
|---|---|---|
| RAW | 265 | 7.83% |
| Moderate CLAHE (clip 2.0) | 3,120 | 92.17% |
| Strong CLAHE (clip 3.0) | 0 | 0.00% |

### 5.3 Statistics of the two measures

| | Global std | Mean local std |
|---|---|---|
| Min | 34.89 | 7.20 |
| 5% | 36.09 | 7.47 |
| 25% | 38.03 | 7.76 |
| Median | 41.96 | 8.40 |
| 75% | 45.62 | 9.83 |
| 95% | 51.75 | 15.57 |
| Max | 54.14 | 19.71 |
| Mean (sd) | 42.29 (4.78) | 9.66 (2.95) |

### 5.4 Finding

v1 does not adapt on this sequence. The minimum global std (34.89) sits just under the 35 gate, so the global test almost never separates frames, and the strong tier is unreachable because the 25 threshold lies about 10 below the minimum. The 265 RAW frames equal the number of frames with local std >= 15, so v1 reduced to a local-only gate at 15, and 92% of frames received CLAHE with settings identical to the fixed-CLAHE baseline. For pairs where both frames are moderate, the preprocessed images are identical to the fixed-CLAHE condition.

A plausible reason is that global std here mostly reflects large-scale illumination (vignetting, lighting falloff, backscatter gradients) and not fine-scale contrast. That explanation is not tested.

## 6. Adaptive v2 (local-contrast gate only)

### 6.1 How it was defined

v2 was defined after v1's decision distribution and the per-frame statistics were seen, and before any matching results were seen. The rule was fixed in advance of looking at the histogram: the threshold would be the minimum of the per-frame histogram of mean local std between the low cluster (around 8) and the high cluster (around 12 to 16), and no threshold would be set if no valley existed.

Histogram of mean local std (1-unit bins):

| Bin | Frames |
|---|---|
| (7, 8] | 1,282 |
| (8, 9] | 1,073 |
| (9, 10] | 199 |
| (10, 11] | 42 |
| (11, 12] | 89 |
| (12, 13] | 147 |
| (13, 14] | 94 |
| (14, 15] | 194 |
| (15, 16] | 124 |
| (16, 17] | 24 |
| (17, 18] | 15 |
| (18, 19] | 31 |
| (19, 20] | 71 |

The minimum between the clusters is the (10, 11] bin, so the threshold is its midpoint, T = 10.5. The valley is modest (42 frames against 199 below and 89 above), not a clean gap. Moving T to 10 or 11 would change the assignment of only a few dozen frames.

The high-contrast frames are not scattered. By segment of 200 frames (frame ID // 200):

| Segment | Frames | v1 RAW | Median local std | Median global std |
|---|---|---|---|---|
| 0 | 200 | 144 | 18.36 | 46.14 |
| 1 | 200 | 0 | 8.78 | 48.07 |
| 2 | 200 | 0 | 7.72 | 45.36 |
| 3 | 200 | 0 | 8.56 | 50.92 |
| 4 | 200 | 0 | 8.75 | 49.17 |
| 5 | 200 | 0 | 8.44 | 43.88 |
| 6 | 200 | 0 | 7.96 | 36.56 |
| 7 | 200 | 0 | 7.52 | 36.60 |
| 8 | 200 | 0 | 7.62 | 38.24 |
| 9 | 200 | 0 | 7.76 | 37.83 |
| 10 | 200 | 0 | 7.92 | 37.63 |
| 11 | 200 | 0 | 7.52 | 37.61 |
| 12 | 200 | 0 | 8.26 | 39.79 |
| 13 | 200 | 0 | 9.57 | 41.31 |
| 14 | 200 | 0 | 13.41 | 42.40 |
| 15 | 200 | 121 | 15.17 | 42.45 |
| 16 | 185 | 0 | 12.52 | 43.06 |

### 6.2 Rule

- Mean local std >= 10.5: keep RAW.
- Otherwise: moderate CLAHE, clip limit 2.0, tile grid 8 x 8.
- No global statistic and no strong tier.

### 6.3 Decision distribution (3,385 frames)

| Mode | Frames | Share |
|---|---|---|
| RAW | 808 | 23.87% |
| Moderate CLAHE (clip 2.0) | 2,577 | 76.13% |
| Strong CLAHE (clip 3.0) | 0 | 0.00% |

This is inside the 790 to 830 range expected from the histogram before the run. v2 changes roughly 525 frames relative to v1: those with local std between about 10.5 and 15, mostly in the transition segments.

### 6.4 Why there is no strong tier

The histogram has a single low cluster with no valley or second low mode, and the lowest local std is 7.2. Any strong-tier cutoff would split that cluster by about half a grey level, which is within exposure and noise variation, and there is no evidence that clip 3.0 beats clip 2.0 on those frames. A strong tier was therefore not added. If stronger enhancement is of interest, the clean test is a separate fixed CLAHE condition with clip 3.0 on all frames, which needs no gating thresholds. Any three-tier method built afterwards would be a new version (v3) and must be reported as post-hoc.

## 7. Implementation

| File | Purpose |
|---|---|
| `adaptive_clahe.py` | Thresholds as module constants, `decide`, `decide_v2`, `enhance`, `apply_adaptive_clahe`, `apply_adaptive_clahe_v2`, `config`, `config_v2`. Importable by other scripts. |
| `aqualoc_sequence6_frontend.py` | Original script plus `--adaptive`, `--adaptive-version {1,2}`, `--decisions-only`, `--baseline-csv`. |
| `test_adaptive_clahe.py` | Asserts on decision logic, determinism, RAW passthrough, and equality of moderate output with fixed CLAHE. |

Outputs of an adaptive run (separate folder):

- `frame_decisions.csv` (per-frame statistics, mode, clip limit, tile grid, statistics time)
- `decision_distribution.json`
- `adaptive_pair_metrics.csv` (pair-level, same columns as the baseline)
- `adaptive_summary.csv` (per detector and stride means, plus pair and `ok` counts)
- `run_metadata.json` (library versions, thresholds, distribution, `ransac_exceptions`)
- `integrity_report.json`

Integrity checks: pair counts per detector match the expected number; same pair set and same columns as the baseline CSV; mode labels are valid; no baseline CSV in the output folder; and for pairs where both frames are RAW or both are moderate, keypoint counts and ratio-match counts reproduce the baseline exactly. The last check covers only deterministic count columns, not RANSAC output.

Commands (v2 shown):

```
python aqualoc_sequence6_frontend.py --image-dir <images> --adaptive --adaptive-version 2 --decisions-only
python aqualoc_sequence6_frontend.py --image-dir <images> --adaptive --adaptive-version 2 --baseline-csv <baseline feature_pair_metrics.csv>
```

Use the same `--max-frames`, `--max-features` and `--pair-strides` as the baseline run.

## 8. Constraints and limitations

1. **Calibrated exclusively to Sequence 06.** Every threshold, and in particular the v2 gate T = 10.5, was derived from the intensity statistics of this one sequence. They will not transfer to other AQUALOC sequences, other cameras or other lighting without recalibration. The v1 thresholds were not derived from the data and happened to be poorly placed for it.
2. **Post-hoc definition.** v2 was defined after seeing v1's decision distribution. It was defined before any matching result was seen. v1 and v2 must both be reported as separate conditions.
3. **Effective sample size.** The high-contrast frames come from a small number of contiguous stretches, so frames are strongly autocorrelated. Frame-level significance tests will overstate certainty. Use block resampling by segment or per-segment reporting.
4. **Correlation is not a mechanism.** The statistics show that the stretches differ in local contrast. Whether skipping CLAHE there improves matching is a hypothesis for the benchmark to test. High local std could also reflect a different scene or distance.
5. **No border mask.** Dark corners or borders would lower the local std. None was applied.
6. **Epipolar error is a proxy.** It comes from an uncalibrated fundamental matrix and is not the Sampson error. The calibrated metrics belong to the other scripts.
7. **Timing excludes preprocessing.** Statistics time is logged per frame, but CLAHE time is not included in the pair timings of any condition.
8. **OpenCV version.** OpenCV 5.0 builds lack `cv2.AKAZE_DESCRIPTOR_MLDB`, which the script requires (tested on 4.10 contrib and 4.13). On OpenCV 4.13.0, `findFundamentalMat` with FM_RANSAC raises an assertion when the matched points in both frames are identical (zero-motion matches), which crashed an adaptive run. The script now catches this error, records the pair with status `fundamental_matrix_failed`, and counts them in `ransac_exceptions` in the metadata. RANSAC settings are unchanged. Such pairs receive zero inliers and are treated as failures, so the baseline and all adaptive runs must be produced with the same OpenCV version, and the number of affected pairs should be reported per condition.
9. **Tested on synthetic data only.** The pipeline and integrity checks were verified on generated frames. Unreadable images in the middle of the sequence and full-scale behaviour were not tested.

## 9. Status and next steps

| Condition | Decisions computed | Benchmark run | Results analysed |
|---|---|---|---|
| RAW (baseline) | not applicable | done (per the user's baseline run) | not in this document |
| Fixed CLAHE (baseline) | not applicable | done (per the user's baseline run) | not in this document |
| Adaptive v1 | yes | not run | no |
| Adaptive v2 | yes | not run | no |

Next steps:

1. Confirm the baseline CSV was produced with the same OpenCV version as the adaptive runs, or rerun it.
2. Run the v2 benchmark with `--baseline-csv`, and check `all_passed` in `integrity_report.json` and `ransac_exceptions` in the metadata before reading any metric.
3. Count the independent RAW stretches in `frame_decisions.csv` to set the effective sample size.
4. Import `apply_adaptive_clahe_v2` into the calibrated and mutual-matching scripts for the remaining Step 6 metrics.
5. Optional: a fixed clip 3.0 condition before considering any strong tier.

## 10. Suggested wording for the method statement

> Adaptive CLAHE selects, for each frame independently, between no enhancement and CLAHE (clip limit 2.0, 8 x 8 tiles) using the mean local standard deviation of intensity over an 8 x 8 grid, with a gate at 10.5. The gate was set at the minimum of the per-frame histogram of this statistic on AQUALOC Sequence 06, after inspection of an earlier two-statistic version (v1) and before any matching results were examined. The gate is calibrated exclusively to this sequence and requires recalibration for other data. No threshold was tuned against evaluation outcomes.
