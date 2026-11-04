# Paper-ready findings and limitations (before AP bootstrap output)

## Suggested title
**Reliability of Underwater Feature Correspondence Under Fixed and Adaptive Contrast Enhancement: A Statistical Study on AQUALOC Sequence 06**

## Evaluation summary

- Dataset: AQUALOC archaeological-site Sequence 06; 3,385 grayscale frames, 968 × 608 px.
- Original frontend benchmark: 60,774 pair records comparing RAW and fixed CLAHE (ORB, AKAZE-MLDB, SIFT; strides 1, 5, 20).
- Existing calibrated reference/mutual benchmark: 14,365 reference pairs × 3 detectors × 2 original preprocessing modes, 86,190 records per evaluation pipeline; Adaptive v2 adds 43,095 reference and 43,095 mutual records. Do not mix these different record totals as though they were all unique observations.
- Adaptive v2 decision: mean local standard deviation ≥ 10.5 retains RAW; otherwise use pre-defined CLAHE levels according to the original `adaptive_clahe.py` rules. On Sequence 06, 808 frames (23.87%) RAW, 2,577 (76.13%) moderate CLAHE, 0 strong CLAHE.
- Threshold 10.5 selected after inspecting Sequence 06's local-std histogram and v1 decision distribution; v2 is **exploratory and sequence-specific**.
- Reference-based results at a 2 px Sampson threshold and minimum eight reference inliers (or eight mutual reference inliers in mutual evaluation).

## Fixed CLAHE vs adaptive v2: reference (non-mutual), all pairs

| Detector | Fixed F1 | Adaptive F1 | Fixed AP (absolute count) | Adaptive AP (absolute count) |
|---|---:|---:|---:|---:|
| ORB | 0.886394 | 0.882088 | 0.930927 | 0.926415 |
| AKAZE-MLDB | 0.611111 | 0.563557 | 0.828474 | 0.814632 |
| SIFT | 0.858618 | 0.852941 | 0.944442 | 0.935690 |

## Fixed CLAHE vs adaptive v2: mutual, all pairs

| Detector | Fixed F1 | Adaptive F1 | Fixed AP (count) | Adaptive AP (count) |
|---|---:|---:|---:|---:|
| ORB | 0.841633 | 0.837581 | 0.918386 | 0.912467 |
| AKAZE-MLDB | 0.547755 | 0.496389 | 0.815782 | 0.796251 |
| SIFT | 0.879668 | 0.867317 | 0.943451 | 0.932795 |

## Local (≤100 frames) and nonlocal (≥200 frames), mutual F1

| Detector | Fixed local | Adaptive local | Fixed nonlocal | Adaptive nonlocal |
|---|---:|---:|---:|---:|
| ORB | 0.934295 | 0.929147 | 0.629630 | 0.629630 |
| AKAZE-MLDB | 0.737143 | 0.677323 | 0.085603 | 0.085603 |
| SIFT | 0.975758 | 0.960799 | 0.696398 | 0.687609 |

## Paired 10-keyframe-anchor bootstrap for mutual F1 (1,000 replicates)

| Detector | All pairs delta F1 [95% CI] | Local delta F1 [95% CI] | Nonlocal delta F1 [95% CI] |
|---|---|---|---|
| ORB | -0.00405 [-0.01336, +0.00138] | -0.00515 [-0.01624, +0.00171] | 0 [0,0] |
| AKAZE-MLDB | -0.05137 [-0.10383, -0.00768] | -0.05982 [-0.11733, -0.00638] | 0 [0,0] |
| SIFT | -0.01235 [-0.03171, -0.00117] | -0.01496 [-0.03200, -0.00234] | -0.00879 [-0.07880, +0.00943] |

## Frontend control & timing

- Original and same-implementation fixed control: zero ratio/inlier mismatches for ORB and AKAZE; SIFT differs in 42 pair ratio-match counts and 40 inlier counts, but has the same binary success classification.
- At a harmonized `>=8 inliers` definition, adaptive vs fixed success rates: ORB 98.766% vs 98.766%; AKAZE 86.050% vs 86.208%; SIFT 99.447% vs 99.447%.
- Average RANSAC inliers adaptive minus fixed: ORB -47.96 (95% frame-block CI [-86.15, -18.47]); AKAZE -133.61 ([-223.60, -56.88]); SIFT -121.64 ([-203.59, -52.66]); 1,000 replicates and 100-frame blocks.
- All 16 lost AKAZE frontend successes occur among 40 mixed RAW–CLAHE pairs at stride 20. Strides 1/5: no mixed-pair success losses. This is a specific sequence-bound observation, not evidence of 16 independent events.
- Preprocessing mean per frame: RAW 0.00126 ms, fixed CLAHE 0.36270 ms, adaptive v2 5.65509 ms; 10,155 timing observations each. Adaptive decision stage ~5.324 ms, applying stage ~0.330 ms. Preprocessing-only, not whole-SLAM runtime; 3 repetitions on a single machine.

## Cautions necessary for rigorous manuscript

1. **Do not claim that Adaptive v2 improves F1, AP, trajectory accuracy or CPU efficiency**: it reduces F1/AP point estimates and increases preprocessing cost compared with fixed CLAHE.
2. AP bootstrap results are still **pending** until `paired_ap_bootstrap.py` is run on the actual pair records and its baseline AP check passes. Do not invent AP confidence intervals.
3. Bootstrap samples based on the first-frame keyframe anchor only partially accommodate shared-frame dependence. Include block-length sensitivity and avoid overconfident significance assertions.
4. Do not overstate COLMAP-derived overlap labels as independently verified physical ground truth. Sim(3) equivalence between two COLMAP exports only validates internal reference consistency; it is not independent pose accuracy evidence.
5. No validated ATE, RPE, map accuracy, or end-to-end SLAM improvement is established by these data.
6. v2 selected on Sequence 06 statistics, evaluated on the same sequence; test additional sequences to support transferability.
7. Multiple detectors, scopes and metrics were assessed. Confidence intervals are descriptive/exploratory unless multiple-testing control or pre-registered hypotheses are established.
8. The source dataset, exact camera calibration/pose convention and hashing of code/results should be documented with the saved manifest before submission.

## Draft conclusion

Fixed CLAHE substantially increased underwater feature matching performance over unenhanced imagery across ORB, AKAZE-MLDB and SIFT on AQUALOC archaeological Sequence 06. Although image-contrast-driven Adaptive CLAHE v2 preserved most pairwise fundamental-matrix-estimation success, it reduced correspondence density, lowered reference-based mutual F1 and AP point estimates, and imposed higher preprocessing overhead. The strongest local performance deterioration occurred for AKAZE-MLDB, while a small SIFT decline was also observed. These results highlight that image-wise contrast selection alone is insufficient to improve underwater correspondence reliability and that temporal consistency and decision-function cost are relevant design considerations. The findings are confined to this sequence and this feature-matching protocol; broader conclusions require independent sequences and downstream visual SLAM evaluation.
