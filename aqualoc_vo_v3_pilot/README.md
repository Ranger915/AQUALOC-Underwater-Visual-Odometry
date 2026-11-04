# AQUALOC Sequence 06 VO V3 pilot — before the full run

This is a controlled pilot **for assessing** whether gap-aware arbitrary translation scaling improves trajectory consistency. It cannot guarantee a lower ATE. It **does not** read COLMAP during VO estimation, and cannot reconstruct metric scale from monocular images alone.

## Main change

V2 forces **one arbitrary translation unit per accepted anchor update**, even if the accepted anchor spans 1, 3, or 20 original frames. V3 makes the translation magnitude policy explicit:

* `--step-scale unit`: V2-style unit translation per accepted keyframe (baseline; other V3 logic identical).
* `--step-scale gap`: translation magnitude = frame separation; a **constant-speed prior**.
* `--step-scale sqrt_gap`: optional intermediate prior.

These are assumptions, not estimated metric velocities. A gap of 10 frames does not imply the camera moved exactly 10 times farther, particularly with rotation, changing speed or loops. Tracking/pose acceptances remain unchanged between unit and gap runs, so comparison isolates the translation-length policy.

## Procedure (PowerShell)

Unzip `aqualoc_vo_v3_pilot.zip` directly inside `archaeo_sequence_6_raw_data`, so the script is at `aqualoc_vo_v3_pilot/slam_engine_sequence6_v3.py`.

```powershell
$root = "$env:USERPROFILE\Downloads\archaeo_sequence_6_raw_data"
$py = "$env:USERPROFILE\aqualoc-venv\Scripts\python.exe"
$engine = "$root\aqualoc_vo_v3_pilot\slam_engine_sequence6_v3.py"

# Use an intermediate 600-frame pilot, NOT the full 3385 frames.
& $py $engine --root $root --max-frames 600 --step-scale unit --out "$root\paper_research_draft\vo_pilot_unit"
if ($LASTEXITCODE -ne 0) { throw "Unit pilot failed" }
& $py $engine --root $root --max-frames 600 --step-scale gap --out "$root\paper_research_draft\vo_pilot_gap"
if ($LASTEXITCODE -ne 0) { throw "Gap pilot failed" }
```

Before evaluation, **verify from the official dataset documentation** whether the trajectory row's `tx ty tz` denotes camera centers (`twc`) or the translation of a world-to-camera pose (`tcw`). The numeric layout alone does not prove this. If `twc`:

```powershell
& $py "$root\aqualoc_vo_v3_pilot\evaluate_pilot.py" --root $root --reference-convention twc
```

Otherwise use `--reference-convention tcw`. The evaluator uses all reference-indexed frames but includes only estimates with a recent accepted update (default maximum 5 frames old) and **requires the same frame set** for both candidates. It fits the Sim(3) transform on the earlier 60% and reports held-out error on later 40%. This removes the in-sample three-point-fit failure seen before, but the two periods are still correlated and reference is COLMAP-derived, not independently measured ground truth.

Outputs: `paper_research_draft/vo_pilot_comparison/pilot_summary.csv`, `.json` and held-out-error PNGs.

## Scientific requirements before paper claims

1. Check reference convention against authoritative documentation, not whichever convention yields lowest ATE.
2. Compare results on the **same reference frames**, report denominators, coverage, and stale/held pose handling.
3. Examine rotation-only spans, tracking failures, outliers, and held-out trajectory shapes.
4. Favor a policy only if it improves errors **consistently** without losing coverage. No guaranteed winner.
5. If tuning using Sequence 06, reserve another sequence for independent validation; do not call this blind held-out validation.
6. Avoid interpreting global Sim(3) fit scale as metric motion, or calling visual odometry full SLAM.
7. For real metric-scale results, use calibrated aligned depth + PnP, visual-inertial metric scale, or another measured scale source; do not feed COLMAP trajectory displacements into the estimator.
