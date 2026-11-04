# AQUALOC Sequence 06 — Underwater Feature Correspondence and Visual Odometry

**A calibrated, reproducibility-focused study of contrast enhancement, feature matching, geometric pose validation, and trajectory reliability in underwater imagery.**

> **Research scope:** Experiments reported here were conducted on **AQUALOC Archaeological Sequence 06**. The processing architecture can be adapted to other calibrated sequences, but cross-sequence performance has **not** been experimentally established. This repository is a research benchmark and experimental monocular visual-odometry (VO) pipeline—not a complete SLAM system or an operational autonomous navigation stack.

## Overview

Underwater imagery is affected by low contrast, nonuniform illumination, turbidity, and repetitive visual texture. These conditions can affect correspondence estimation and downstream camera-motion recovery in different ways. This project systematically evaluates those stages rather than assuming that a higher feature-match count automatically improves localization.

We investigated **ORB, AKAZE-MLDB, and SIFT**, tested raw images against **fixed CLAHE** and a lightweight **frame-adaptive CLAHE policy**, and evaluated correspondences using calibrated reference geometry and mutual matching. We then developed a monocular VO diagnostic pipeline using **descriptor matches and forward–backward KLT optical flow**, essential-matrix estimation, cheirality validation, and quality-based pose-source selection.

### Research contributions

- A full-sequence frontend benchmark, progressing from initial uncalibrated matching to calibration-aware geometric verification.
- Paired, fixed-versus-adaptive enhancement experiments with precision, recall, F1, average precision (AP), match/inlier counts, and preprocessing overhead.
- Reference-pair evaluation on **14,365 unique unordered pairs** from **170** COLMAP-derived keyframes, including local and nonlocal strata.
- Bootstrap uncertainty estimates and block-size sensitivity analyses; negative and inconclusive findings are retained.
- Iterative VO diagnostics (initial baseline, adaptive anchoring, dual descriptor/flow candidates, cheirality threshold and selection-policy ablations).
- A frozen **3,385-frame** VO run, including failure reasons, tracking coverage, and chronological held-out trajectory-consistency evaluation.
- A numerical reproducibility audit: **43/43 targeted source-data checks passed**, alongside SHA-256 source-file manifests. This does **not** certify every possible implementation or publication claim.

## Dataset and calibration

| Item | Sequence 06 |
|---|---:|
| Grayscale images | **3,385** |
| Image resolution | **968 × 608** |
| Image rate | **≈20 Hz** |
| Duration | **≈169.17 s** |
| COLMAP-derived reference keyframes | **170** |
| Keyframe sampling | Every 20 image frames |
| Unique unordered reference pairs | **14,365** |
| Reference-positive overlap pairs | **1,143 (7.96%)** |

The camera model is pinhole with radial–tangential distortion:

```yaml
resolution: [968, 608]
fx: 543.3327734182214
fy: 542.398772982566
cx: 489.02536042247897
cy: 305.38727712002805
# [k1, k2, p1, p2]
distortion: [-0.1255945656257394, 0.053221287232781606, 0.0000994070021080493, 0.00009550660927242349]
```

**Dataset source:** [AQUALOC (official dataset website)](https://www.lirmm.fr/aqualoc/). Obtain raw images, calibration and reference products from the dataset maintainers and follow their distribution and citation requirements. The original dataset is **not included** in the repository unless explicitly indicated.

### Reference-pose provenance

The AQUALOC trajectory export is recorded as `frame tx ty tz qx qy qz qw` (**quaternion order: xyzw**), unlike standard COLMAP `images.txt` ordering. We use `colmap_traj_sequence_6.txt` as the principal Sequence 06 reconstruction-derived reference. The direct XYZ path length (≈31.827 m) agrees with the original AQUALOC publication's reported ≈31.8 m pressure-scaled path length; the alternate `new_archaeo_colmap_traj_sequence_06.txt` is related by an almost exact Sim(3) transform and **must not** be counted as independent ground truth. The exact serialized rotation-frame convention is not certified by an available exporter source, so pose-direction assumptions remain documented in evaluation outputs.

**Interpretation warning:** COLMAP-derived poses are an *approximate reconstruction reference*, not independently measured absolute ground truth. Do not report monocular alignment errors as independent metric SLAM accuracy.

## Headline results

### Mutual reference-based matching

All **14,365** unordered keyframe pairs; reference threshold **2 px Sampson error** and at least **8 mutually matched reference-consistent inliers**. AP shown below ranks pairs by **reference-consistent inlier count**; it is not interchangeable with AP calculated using an inlier fraction score.

| Detector | Preprocessing | F1 | Count-based AP |
|---|---|---:|---:|
| ORB | Raw | 0.2940 | 0.5488 |
| ORB | **Fixed CLAHE** | **0.8416** | **0.9184** |
| ORB | Adaptive CLAHE v2 | 0.8376 | 0.9125 |
| AKAZE-MLDB | Raw | 0.1696 | 0.3576 |
| AKAZE-MLDB | **Fixed CLAHE** | **0.5478** | **0.8158** |
| AKAZE-MLDB | Adaptive CLAHE v2 | 0.4964 | 0.7963 |
| SIFT | Raw | 0.3944 | 0.6536 |
| SIFT | **Fixed CLAHE** | **0.8797** | **0.9435** |
| SIFT | Adaptive CLAHE v2 | 0.8673 | 0.9328 |

**Finding:** Fixed CLAHE substantially improves correspondence retrieval over raw preprocessing at this operating point. The tested adaptive switching policy does not improve overall F1 or count-based AP over fixed CLAHE on Sequence 06; this is a result for the **specific policy and sequence**, not a claim about all adaptive underwater enhancement methods.

### Temporal uncertainty

For all-pairs adaptive-minus-fixed differences, additional paired block-size sensitivity used **400 bootstrap replicates** per tested setting and blocks of **1, 5, 10, 20** reference keyframes:

| Detector | ΔF1 (adaptive − fixed) | Interpretation across tested blocks |
|---|---:|---|
| ORB | −0.00405 | All examined confidence intervals include zero |
| AKAZE-MLDB | −0.05137 | Intervals are negative or touch zero |
| SIFT | −0.01235 | One block-size interval includes zero |

Confidence intervals are **unadjusted** exploratory intervals and cannot fully eliminate temporal or dyadic dependence. Policy development and selection occurred within Sequence 06; the repository does not claim an untouched independent-sequence validation.

### Frozen full-sequence monocular VO

| Metric | Result |
|---|---:|
| Frames processed | **3,385** |
| Attempted frame transitions | **3,384** |
| Accepted motion updates | **1,188 (35.11%)** |
| Freshly accepted COLMAP-indexed frames | **60 / 170 (35.29%)** |
| Low-cheirality-fraction rejections | 1,227 |
| Insufficient-cheirality-inlier rejections | 962 |
| Low-parallax rejections | 7 |
| Descriptor-valid / flow-valid candidates | 1,012 / 356 |
| Both valid | 180 |
| Exclusive additional accepted updates from optical flow | **176** |
| Chronological alignment / held-out fresh poses | **36 / 24** |
| Held-out Sim(3)-aligned positional RMSE | **4.0893 reconstruction units** |
| Held-out median positional error | **3.9457 reconstruction units** |

The trajectory-error values depend on the documented camera-center/`Twc` interpretation of the original AQUALOC export. **They are not independent ground-truth ATE in metres.** The VO trajectory is **monocular and arbitrary-scale**; its gap-aware translation scale is a heuristic, not metric depth recovery. Accepted updates do not guarantee accurate trajectory estimates, and rejected frames may carry stale poses.

**Takeaway:** High calibrated matching F1/AP and frequent descriptor correspondences do not by themselves guarantee reliable cheirality-validated motion or low accumulated trajectory error.

## Methods at a glance

```text
AQUALOC Sequence 06 grayscale frames
                │
       Raw / fixed CLAHE / adaptive-v2
                │
      ORB / AKAZE-MLDB / SIFT
                │
        Descriptor correspondences
                ├──► Frontend match & inlier statistics
                ├──► Calibrated Sampson geometry
                └──► Mutual matching / reference-overlap F1 & AP

Frozen exploratory VO (ORB + fixed CLAHE):
                │
       Descriptor matching ──────┐
                                 ├──► Independent essential matrix
       Forward–backward KLT ────┘      and pose recovery per source
                                            │
                                Cheirality & parallax gates
                                            │
                                Quality-based valid-pose choice
                                            │
                                Monocular pose accumulation
                                            │
                              Tracking, coverage & held-out drift
```

### Fixed VO configuration

- ORB with up to **1,500** features; fixed CLAHE preprocessing.
- Descriptor ratio threshold **0.75**; essential-matrix RANSAC **1.5 px**.
- Optical flow: up to **1,800** points; forward–backward tolerance **1.5 px**.
- At least **15 essential inliers**, **10 cheirality inliers**, and **0.06** minimum cheirality fraction.
- Minimum parallax **0.1°**; maximum anchor gap **20** frames.
- Evaluate descriptor and optical-flow candidates separately and select with `--source-policy quality`.
- Use `--step-scale gap` for **arbitrary-unit** frame-gap-dependent motion, not a metric displacement estimate.

## Repository organization

The following is the **recommended layout** for a public repository. The actual files you upload determine which paths exist; do not assume every experimental artifact has already been released.

```text
.
├── README.md
├── LICENSE                         # Add your chosen code license, if any
├── requirements.txt               # Freeze dependencies used for reproduction
├── src/
│   └── slam_engine_sequence6_v3.py
├── scripts/
│   ├── evaluate_full_v3.py
│   └── ...                        # Include the actual benchmarking/audit scripts
├── config/
│   └── ...                        # Camera/IMU calibration + run settings
├── results/
│   ├── frontend/
│   ├── calibrated/
│   ├── mutual_matching/
│   ├── bootstrap/
│   └── vo/
├── figures/                       # Selected, verified paper figures
├── paper/
│   └── AQUALOC_Sequence06_Final_Illustrated_Manuscript.docx
└── manifests/                     # SHA-256 and validation reports
```

If you preserve the original local `archaeo_sequence_6_raw_data/` directory instead of reorganizing files, adjust the paths in the commands below accordingly.

## Quick start (Windows PowerShell)

These commands apply to the **original Sequence 06 directory structure** used during the experiments. They require the actual input images and the V3 engine file, which are not supplied by this README alone.

```powershell
$root = "$env:USERPROFILE\Downloads\archaeo_sequence_6_raw_data"
$py = "$env:USERPROFILE\aqualoc-venv\Scripts\python.exe"
$engine = "$root\aqualoc_vo_v3_pilot\slam_engine_sequence6_v3.py"

# Confirm that required inputs exist
if (!(Test-Path $py)) { throw "Python virtual environment not found: $py" }
if (!(Test-Path $engine)) { throw "VO engine not found: $engine" }
if (!(Test-Path "$root\raw_data\images_sequence_6")) { throw "Sequence 06 images not found" }

& $py -u $engine `
    --root $root `
    --max-frames 0 `
    --algorithms ORB `
    --preprocessings clahe `
    --max-features 1500 `
    --flow-points 1800 `
    --flow-fb-px 1.5 `
    --ratio 0.75 `
    --ransac-px 1.5 `
    --min-essential-inliers 15 `
    --min-cheirality-inliers 10 `
    --min-cheirality-fraction 0.06 `
    --min-parallax-deg 0.1 `
    --max-anchor-gap 20 `
    --step-scale gap `
    --source-policy quality `
    --out "$root\paper_research_draft\vo_full_sequence_v3"

if ($LASTEXITCODE -ne 0) { throw "VO run failed" }
```

Full-sequence evaluation, using the evaluator package from the study:

```powershell
$script = "$root\aqualoc_full_sequence_eval\evaluate_full_v3.py"
if (!(Test-Path $script)) { throw "Evaluation script not found: $script" }
& $py $script --root $root --reference-convention twc --min-train 10 --min-test 5
if ($LASTEXITCODE -ne 0) { throw "Evaluation failed" }
```

The chronological split actually used in the frozen run contained **36 train / 24 held-out fresh poses**. Inspect CLI help and output metadata before assuming that all options or paths apply to an alternate code layout.

### Python dependencies and tested environment

The tested analysis environment was **Windows 11**, **Python 3.13.14**, **OpenCV 4.11.0 (contrib build)**, **NumPy 2.5.4**, **Pandas 3.1.1**, and **Matplotlib 3.11.2**, on an **AMD Ryzen 7 6800HS / NVIDIA RTX 3050 Laptop GPU / 16 GB RAM** machine. The full audit captured OpenCV build information. For a reproducible installation, **publish the exact tested `pip freeze`** as a lock file rather than treating the following illustrative command as a verified lock:

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install numpy pandas matplotlib opencv-contrib-python
```

Some feature detectors require the contrib build; OpenCV constructors can differ across releases. Check the frozen code and environment before upgrading OpenCV.

## Reproducibility and evaluation integrity

1. **No reference leakage into online VO:** COLMAP references are used for offline labels/evaluation, not for per-frame VO candidate selection. Parameter tuning and exploratory ablations were nonetheless performed on Sequence 06 and must not be described as independent testing.
2. **Report coverage with errors:** Distinguish fresh pose updates from carried-forward positions. The held-out RMSE uses 24 fresh reference-indexed poses, not all 3,385 frames.
3. **State metric definitions:** Pair-level AP is based on the *specified reference-consistent inlier count* unless explicitly labeled otherwise; mutual matching, thresholds, scope, and minimum inlier count must accompany each F1/AP result.
4. **Preserve provenance:** Publish the frozen engine and evaluator versions, run arguments, CSV schemas, hashes, and audit reports. The targeted numeric audit reproduced **43/43 checked items**; it did not prove complete hardware-level determinism.
5. **Avoid invented outputs:** Parameter-response heatmaps are **not** underwater occupancy/cost maps. This project does **not** provide a measured environmental point cloud, depth map, occupancy grid, or map-based SLAM reconstruction.
6. **Respect dataset rights:** Avoid committing large raw image archives or datasets whose redistribution terms are unclear; link to the original source instead.

## Figures and manuscript

The research paper contains selected feature maps, calibrated-response heatmaps, mutual-matching results, bootstrap comparisons, cheirality/selection diagnostics, reference coverage, rejection distributions, parallax diagnostics, and trajectory projections. Figures were checked against their contents; an automatically selected duplicate AP plot and an older pilot image mislabeled as a full tracking timeline were excluded from the illustrated manuscript.

Add figures only after reviewing captions and preserving their actual study conditions. If included, a convenient layout is `figures/` with descriptive filenames and `paper/` for the illustrated Word manuscript. Broken README image links should not be added before the files exist on GitHub.

## Limitations and future directions

- **Single-sequence experimental evidence:** The pipeline is **calibration-portable in principle**—camera intrinsics, distortion, time metadata, and, when used, camera–IMU transforms can be replaced for other sequences. This is not measured cross-sequence generalization.
- **Reference qualification:** The AQUALOC camera poses derive from offline COLMAP reconstruction and pressure-assisted scaling. Exact export rotation direction remains inferred; alternative local exports are not independent ground truth.
- **Monocular scale and drift:** Gap-aware translation scaling does not resolve metric scale ambiguity, and tracking acceptance does not guarantee geometric accuracy.
- **Statistical scope:** Bootstrap results use unadjusted confidence intervals, correlated image pairs, and sequence-specific policy development.
- **No full SLAM:** There is no established loop-closure optimization, global bundle adjustment, or validated 3D navigation map in the VO pipeline.

Possible future work includes multi-sequence calibration transfer, a fully specified frame-convention audit, more robust uncertainty resampling, and systematic diagnosis of cheirality failures and stale-track handling.

## Citation and attribution

If you use the dataset, **cite the original AQUALOC publication and follow its official citation guidance** at [the AQUALOC website](https://www.lirmm.fr/aqualoc/). For the experimental findings and code in this project, cite the accompanying manuscript when a citable publication or archival identifier becomes available; **no DOI is claimed here**.

## Copyright and Intellectual Property

**Copyright © 2026 RAAGHAVAN MOOKIAH BARATHKUMAR. All Rights Reserved.**

**Copyright Holder:** RAAGHAVAN MOOKIAH BARATHKUMAR
**Contact:** [raaghavanbkm@gmail.com](mailto:raaghavanbkm@gmail.com)

The original source code, experimental implementations, documentation, research figures, and original written content in this repository are claimed as the intellectual property of RAAGHAVAN MOOKIAH BARATHKUMAR, subject to any applicable third-party or institutional rights.

No license is granted to reproduce, modify, redistribute, sublicense, or commercially exploit the original copyrighted material without prior written permission from the copyright holder, except as permitted by applicable law or platform terms.

Academic researchers may reference the published findings with appropriate attribution. Permission to reproduce protected code, figures, or substantial portions of the manuscript must be obtained separately.

### Third-Party Dataset Attribution

The AQUALOC dataset and its original materials remain the property of their respective rights holders and are governed by their applicable licensing and usage terms. Ownership claims in this repository apply only to original contributions.

### Permission Requests

For licensing, collaboration, reproduction, or commercial-use requests, contact:

**RAAGHAVAN MOOKIAH BARATHKUMAR**
[raaghavanbkm@gmail.com](mailto:raaghavanbkm@gmail.com)

---

*Research focus: rigorous evaluation of the gap between underwater correspondence quality, geometric pose recoverability, and long-horizon trajectory reliability.*
