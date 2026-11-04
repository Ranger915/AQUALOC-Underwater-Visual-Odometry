# AQUALOC Sequence 06 — Paper Research Draft Figure Suite

**What it does:** creates 300-DPI PNG figures in a SEPARATE directory:

`C:\Users\Raaghav\Downloads\archaeo_sequence_6_raw_data\paper_research_draft\figures\`

The scripts only READ your existing CSV/image/trajectory inputs. They write figures and summaries under `paper_research_draft` and never modify source data.

## Installation (PowerShell)

Unzip `paper_research_draft_figures.zip` **inside** `C:\Users\Raaghav\Downloads\archaeo_sequence_6_raw_data\` so that the scripts appear under `...\paper_research_draft\scripts\`.

```powershell
$root = "$env:USERPROFILE\Downloads\archaeo_sequence_6_raw_data"
$py = "$env:USERPROFILE\aqualoc-venv\Scripts\python.exe"
& $py -m pip install numpy pandas matplotlib
```

If your current virtual environment already has working AKAZE/ORB/SIFT, **do not change its OpenCV installation**; only install missing `pandas` / `matplotlib` / `numpy` as needed.

## Run all existing-data comparisons

```powershell
$root = "$env:USERPROFILE\Downloads\archaeo_sequence_6_raw_data"
$py = "$env:USERPROFILE\aqualoc-venv\Scripts\python.exe"
& $py "$root\paper_research_draft\scripts\run_all.py" --root $root --frame-a 100 --frame-b 120
explorer.exe "$root\paper_research_draft\figures"
```

The batch runner attempts all four modules and reports any missing inputs. `01_slam_ate_trajectories.py` is separate because a *real SLAM-estimated trajectory* is mandatory.

## Run dedicated scripts individually

```powershell
& $py "$root\paper_research_draft\scripts\02_feature_maps_matches.py" --root $root --frame-a 100 --frame-b 120
& $py "$root\paper_research_draft\scripts\03_cost_maps_heatmaps.py" --root $root
& $py "$root\paper_research_draft\scripts\04_adaptive_vs_fixed.py" --root $root
& $py "$root\paper_research_draft\scripts\05_publication_bar_charts.py" --root $root
```

### 01 — TRUE SLAM vs reference ATE / trajectories

**Important:** Your two pre-existing COLMAP trajectory exports align almost perfectly after Sim(3). That result measures *equivalence of exports*, **not SLAM accuracy**. You cannot obtain real SLAM-vs-reference ATE until the SLAM pipeline produces a separate estimated trajectory.

Accepted estimated/reference pose formats:

* **TUM**: `timestamp tx ty tz qx qy qz qw`, whitespace or comma separated; timestamps are seconds by default (configurable)
* **frame_tquat**: `frame_id tx ty tz qx qy qz qw`, whitespace or comma separated
* **csv**: named columns `frame_id` OR `timestamp`, plus `tx,ty,tz,qx,qy,qz,qw`
* **KITTI**: one 3x4 matrix (12 numbers) per line; sequential frame IDs 0..N−1. Matrix translation convention must be specified.

For each input, provide its **pose convention**: `twc` (the translation is camera position expressed in world coordinates, such as camera-to-world matrix translation) or `tcw` (world-to-camera translation, converted via `-R^T t`). This is critical for COLMAP: world-to-camera `t` is NOT the camera center. Confirm actual convention from the SLAM exporter and reference file. Camera coordinate vs IMU coordinates and timestamp offsets/extrinsics must also be reconciled before interpreting ATE.

**Example: TUM estimated + TUM reference, metric RGB-D SLAM, SE(3) alignment**:

```powershell
$est = "$root\slam_outputs\estimated_tum.txt"       # replace with ACTUAL SLAM output
$ref = "$root\reference_outputs\reference_tum.txt"  # replace with ACTUAL trusted reference
& $py "$root\paper_research_draft\scripts\01_slam_ate_trajectories.py" `
  --root $root --estimated $est --reference $ref `
  --est-format tum --ref-format tum --est-convention twc --ref-convention twc `
  --associate timestamp --tolerance 0.02 --alignment se3
```

**Example: frame-indexed SLAM CSV + COLMAP export (after verifying COLMAP column interpretation)**:

```powershell
$est = "$root\slam_outputs\estimated_poses.csv"
$ref = "$root\colmap_traj_sequence_6.txt"
& $py "$root\paper_research_draft\scripts\01_slam_ate_trajectories.py" `
  --root $root --estimated $est --reference $ref `
  --est-format csv --ref-format frame_tquat --est-convention twc --ref-convention tcw `
  --associate frame_id --alignment se3
```

**CAUTION**: The second command is only illustrative. Verify COLMAP file column semantics first: frame IDs, quaternion order and whether pose fields are `Tcw` or camera centers, plus scale and coordinate frames. The source's metric scale is not established independently. The ATE script refuses to run without an estimated and a reference file. It does not produce an ATE plot by comparing the two COLMAP exports automatically.

* For metric RGB-D SLAM: default `--alignment se3`, preserve the real scale.
* For monocular SLAM with unknown scale: `--alignment sim3`; clearly report the estimated scale factor.
* `--associate timestamp` uses `--tolerance` seconds and optional `--time-offset` (added to estimated timestamps).
* Only **translational ATE** is evaluated; no claim of rotation ATE, RPE or full SLAM mapping accuracy.
* No timestamp interpolation is performed. Only matched poses are evaluated; note coverage of original trajectory when interpreting results.

Outputs: `ate_metrics.json`, `ate_per_pose.csv`, 3-view XYZ trajectory overlays, 3D trajectory, ATE by pose, ATE histogram and empirical CDF.

## 02 — Feature maps and match visualization

Reads actual raw Sequence 06 frames and `sequence6_adaptive_v2_full/adaptive_frame_decisions.csv`. Creates keypoint overlays, 8×8 keypoint density maps, and visual matches for all 3 detectors × 3 preprocessing methods. Features and tentative matches are recomputed to visualize geometry; the fundamental-matrix inliers come from a newly estimated `F`, **not COLMAP reference inliers**, and may differ from saved benchmark metrics. Values per example pair are in `visualization_pair_metrics.csv`. The AKAZE fallback checks both `cv2.AKAZE_create` and `cv2.xfeatures2d.AKAZE_create`.

## 03 — Parameter-response heatmaps (aka evaluation 'cost maps')

Uses the **actual calibrated and mutual summary CSV files** to plot F1, precision, recall across Sampson thresholds and minimum verified inliers for all pairs, local pairs and non-local pairs, for ORB, AKAZE-MLDB and SIFT × RAW/fixed CLAHE/adaptive v2. These are **parameter-response surfaces**, **not** spatial navigation cost maps or an occupancy map. A genuine spatial cost map requires registered geometry/depth/poses and a defined cost function.

## 04 — Adaptive vs fixed comparisons

Paired differences (adaptive v2 minus fixed CLAHE) across detectors and scopes for F1/precision/recall, plus AP paired-bootstrap 95% CI (and F1 CI when discoverable). A confidence interval crossing zero is **not** evidence of a significant difference.

## 05 — Publication-quality bar charts

Detector × preprocessing comparison charts for calibrated/mutual F1, precision, recall and count AP at a fixed 2 px and 8 verified-inlier operating point; also local and non-local scope comparisons. The underlying AP scores are *ranking metrics* and not functions of decision minimum when computed from the same scores/labels; the operating-point filter is for consistent selection.

## Non-negotiable scientific caveats

- AQUALOC COLMAP reference is a reconstruction-derived **proxy reference**, not independently validated ground truth.
- Only Sequence 06 has been tested; adaptive threshold was selected from its own image statistics.
- AP and F1 intervals use block bootstrap and should not be treated as validation on new sequences.
- Figure labels do **not** imply robot occupancy mapping, collision-cost maps, actual SLAM trajectory estimation, or visual odometry errors unless those have been independently computed.
- If a CSV field is missing or duplicated, the scripts **fail visibly**, rather than quietly filling in fake data.
