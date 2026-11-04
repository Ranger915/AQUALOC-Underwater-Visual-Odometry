# Sequence 06 — coverage-aware VO trajectory evaluation

This package evaluates **your new Sequence 06 VO V2 trajectories** against a COLMAP reconstruction, and writes PNG plots inside `paper_research_draft/figures/trajectory_evaluation`.

## Essential interpretation

- Your COLMAP files contain `frame_id tx ty tz qx qy qz qw` rows at frames 0, 20, ..., 3380.
- **These rows alone do not establish pose convention.** You MUST decide using dataset documentation whether `tx ty tz` are camera centers in world (`twc`) or world-to-camera translations (`tcw`). Use the explicit `--reference-convention` flag. In `tcw` mode, XYZ is converted to the camera center as `-R_cw.T @ t_cw` using the quaternion `qx qy qz qw`.
- VO V2 CSV `x y z` columns represent camera positions in its arbitrary coordinate frame. Its monocular translation step is arbitrary; global Sim(3) aligns scale, orientation and origin but **cannot remove local scale drift**.
- **COLMAP reconstructed trajectories are reference estimates, not independent measured ground truth.** Name the result "Sim(3)-aligned position error relative to COLMAP", not independent ground-truth SLAM ATE.
- The script preferentially aligns/evaluates **only reference-indexed accepted transitions** (`tracking_success=1`). It reports coverage relative to 170 reference poses. All-associated error, including held/stale failed poses, is separately labeled. A fit requires >=3 accepted reference-indexed estimates, so a 100-frame smoke test may have insufficient matches; full 3385-frame run is normally needed.
- Alignment and error measurement use the same points, so the error is **in-sample**. Report as a best-aligned trajectory residual and do not infer performance from the figure alone. No RPE/rotation accuracy is claimed by this script.
- Treat COLMAP pose units as unverified until dataset documentation confirms metric scale.

## PowerShell

Extract folder `sequence6_trajectory_evaluation` into `C:\Users\Raaghav\Downloads\archaeo_sequence_6_raw_data`.

```powershell
$root = "$env:USERPROFILE\Downloads\archaeo_sequence_6_raw_data"
$py = "$env:USERPROFILE\aqualoc-venv\Scripts\python.exe"
$script = "$root\sequence6_trajectory_evaluation\evaluate_sequence6_vo.py"

# IMPORTANT: choose twc or tcw based on official metadata, not whichever produces smaller error.
& $py $script --root $root --reference-convention twc

explorer.exe "$root\paper_research_draft\figures\trajectory_evaluation"
```

If documentation establishes world-to-camera (`Tcw`) instead, rerun with `--reference-convention tcw`. Do not select the convention post hoc on the basis of ATE.

The script does not alter original research outputs. It produces a JSON manifest, per-configuration CSV values, diagnostic plots, and coverage summaries.
