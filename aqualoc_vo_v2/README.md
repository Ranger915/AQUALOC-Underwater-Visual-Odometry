# AQUALOC Sequence 06 — Calibrated VO Engine v2

This is a **replacement experiment runner**, not an overwrite of `C:\Users\Raaghav\Desktop\deep_sea\slam_engine.py`. It directly reads the **3,385 PNG frames** in `raw_data/images_sequence_6` and saves all results to `paper_research_draft/trajectories_v2`.

## What was changed, and why

1. **Built-in AQUALOC camera calibration:** `(fx,fy,cx,cy)=(543.3327734182214,542.398772982566,489.02536042247897,305.38727712002805)`, radial-tangential distortion `(-0.1255945656257394,0.053221287232781606,9.94070021080493e-05,9.550660927242349e-05)`. No generic guessed FOV.
2. **Read source images directly:** avoids converting the dataset to an MP4 or making FPS assumptions for pose estimation.
3. **ORB/AKAZE/SIFT; raw or fixed CLAHE:** all use 1,500 maximum features and default Lowe ratio 0.75. Optional mutual matching.
4. **Diagnostic decomposition:** saves raw descriptor matches, essential-matrix RANSAC inliers, cheirality survivors, both denominators for ratios, and rotation-compensated median parallax. Unlike the original engine, it does not evaluate cheirality inliers **only** as a fraction of all matches.
5. **Adaptive keyframe anchor:** attempts motion over up to 20 frames if immediate-neighbor motion is inconclusive. Importantly, repeated anchor-based poses are not integrated twice.
6. **No fabricated trajectory motion:** when a pose fails validation, the previous pose is held and the failure is recorded. If an anchor must be replaced without known motion, the gap remains unknown. Flat trajectory sections therefore represent **missing motion**, not successful stationary tracking.
7. **All new CSVs and JSON in separate directory:** baseline and original AQUALOC benchmark data remain untouched.

## Install

Python virtual environment must have `opencv-python`, `numpy` (no pandas required for this script). If OpenCV AKAZE is missing in a custom build, try an environment with modern `opencv-contrib-python`; don't install conflicting OpenCV distributions simultaneously.

Extract the ZIP such that the script is at:

`C:\Users\Raaghav\Downloads\archaeo_sequence_6_raw_data\aqualoc_vo_v2\slam_engine_sequence6_v2.py`

## Run: first 100 frames

```powershell
$root = "$env:USERPROFILE\Downloads\archaeo_sequence_6_raw_data"
$py = "$env:USERPROFILE\aqualoc-venv\Scripts\python.exe"
$script = "$root\aqualoc_vo_v2\slam_engine_sequence6_v2.py"

& $py $script --root $root --max-frames 100
if ($LASTEXITCODE -ne 0) { throw "VO v2 failed; inspect error above." }
```

For a *diagnostic*, try ORB with more inclusive essential-matrix RANSAC threshold (this is a **different experimental configuration**, not a validated improvement):

```powershell
& $py $script --root $root --max-frames 100 --algorithms ORB --preprocessings clahe --ransac-px 2.0 --out "$root\paper_research_draft\trajectories_v2_ransac2"
```

**Only after inspecting tracking diagnostics**, run all frames:

```powershell
& $py $script --root $root --max-frames 0
```

## Outputs

`paper_research_draft/trajectories_v2/`

- `vo_sequence6_orb_raw_v2.csv`
- `vo_sequence6_orb_clahe_v2.csv`
- `vo_sequence6_akaze_raw_v2.csv`
- `vo_sequence6_akaze_clahe_v2.csv`
- `*_keyframes.csv` sampled at frame IDs 0,20,40,...
- `run_summary.json` includes rejection status counts and run settings.

Each CSV has a `status` column: `accepted`, `low_parallax`, `insufficient_essential_inliers`, `insufficient_cheirality_inliers`, `low_cheirality_fraction`, etc.; `tracking_success=1` only for accepted motion. Both `cheirality_ratio_all` and `cheirality_ratio_essential` are recorded, so the source of the original 1–8% ratios can be diagnosed correctly.

The poses are world-to-camera **camera positions in an arbitrary-scale world frame** (camera-to-world `T_wc`). Each accepted anchor-to-current translation is normalized to length 1. This arbitrary step length is *not* physically metric and can accumulate spatially inconsistent drift.

`time_sec` is `sequence_index/20.0` as a nominal timestamp for human inspection, not exact camera timestamps. Use the original image timestamp CSV for accurate timestamp association.

## Study / publishing limitations

- **This is monocular visual odometry, not full SLAM**: no global bundle adjustment, loop closure, map maintenance, or depth integration.
- **No claim of a successful trajectory** is warranted until AQUALOC images are actually run and `tracking_success` / failure modes are inspected.
- Even if more transitions are accepted, **acceptance is not accuracy**. Confirm against an independently supplied pose reference using the correct COLMAP pose convention (invert if `T_cw` rather than `T_wc`), synchronization and keyframe IDs.
- **Sim(3) removes only a single global scale factor**; it does not fix per-frame monocular scale drift, freezes across failures, or lost tracking. Report trajectory coverage and untracked intervals alongside ATE. Do not compute an apparently low ATE from long repeated held poses without flagging the failures.
- The COLMAP trajectory and overlap labels are reconstructions derived from the same dataset and should be described as *COLMAP-derived reference*, not independent externally measured ground truth.
- There is no guarantee this architecture will recover most motion on a low-parallax sequence; better tracking may need optical flow, inertial information, depth, local BA or a mature VIO/SLAM implementation.

## Dependencies and sanity check

```powershell
& $py -c "import cv2,numpy; print('OpenCV',cv2.__version__,'NumPy',numpy.__version__)"
Get-ChildItem "$root\paper_research_draft\trajectories_v2" -Filter '*.csv' | Select-Object Name,Length
```

To display failure breakdown:

```powershell
& $py -c "import json,sys; j=json.load(open(sys.argv[1])); [print(r['algorithm'],r['preprocessing'],r['accepted'],'/',r['frames']-1,r['statuses']) for r in j]" "$root\paper_research_draft\trajectories_v2\run_summary.json"
```
