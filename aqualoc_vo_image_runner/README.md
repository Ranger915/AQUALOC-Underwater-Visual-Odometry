# AQUALOC Sequence 06 — monocular VO from individual PNG images

This runner imports (does not edit) `C:\Users\Raaghav\Desktop\deep_sea\slam_engine.py` and feeds its ORB/AKAZE, descriptor matching, essential matrix, recoverPose, and row-serialization functions with the calibrated AQUALOC frame sequence directly. **No lossy MP4 conversion is needed.**

## Install if necessary

```powershell
$py = "$env:USERPROFILE\aqualoc-venv\Scripts\python.exe"
& $py -m pip install numpy opencv-contrib-python
```

## Run a small smoke test

Extract ZIP into the project root (`archaeo_sequence_6_raw_data`) and run:

```powershell
$root = "$env:USERPROFILE\Downloads\archaeo_sequence_6_raw_data"
$py = "$env:USERPROFILE\aqualoc-venv\Scripts\python.exe"
& $py "$root\aqualoc_vo_image_runner\run_sequence6_vo.py" --root $root --max-frames 100
```

## Full sequence

```powershell
& $py "$root\aqualoc_vo_image_runner\run_sequence6_vo.py" --root $root
```

Outputs in `paper_research_draft\trajectories`: four ORB/AKAZE x raw/CLAHE pose CSVs, four COLMAP-stride-20 keyframe CSVs, JSON metadata and a run manifest.

## Scientific cautions

- This is visual **odometry**, not a full loop-closure SLAM system.
- `scale_mode=monocular_arbitrary`; the engine applies a unit translation on every accepted frame transition. Coordinates **are not metres**.
- With no external scale, **Sim(3)** alignment is required for translation-error comparison; even then, time-varying scale drift remains and differences cannot be treated as metric online localization accuracy.
- Failed recoverPose transitions hold the last pose constant. Check acceptance rates and coverage rather than only RMSE.
- AQUALOC COLMAP reference poses can have different coordinate conventions and orientation formats. Verify these before evaluating ATE. Never compare the two COLMAP exports and call that SLAM ATE.
- Comparisons here use the VO engine's ratio=0.80 defaults, which are **not the same experiments** as the earlier correspondence study with ratio=0.75. Keep these results separate.
- These scripts use original unmodified RGB images and fixed CLAHE 2.0. Adaptive-v2 comparison needs an implementation-faithful adaptation of the existing adaptive preprocessing function.
