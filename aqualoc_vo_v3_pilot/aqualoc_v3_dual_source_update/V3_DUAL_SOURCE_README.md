# AQUALOC Sequence 06 V3 — paired geometric candidate selection

## Installation

Replace ONLY the existing `aqualoc_vo_v3_pilot/slam_engine_sequence6_v3.py` with the provided file, after backing up the original. The Python file is the full script (not a patch). No change to `evaluate_pilot.py` is required to read the standard trajectory fields.

## What changed

- Run descriptor matches AND forward–backward KLT optical flow for the same anchor/current frame pair.
- Estimate essential matrix and recover pose separately for each source.
- Select only geometrically *accepted* candidates. Options: `--source-policy quality` (default) or `--source-policy descriptor_first`.
- Quality scores accepted candidates using cheirality count, fraction of essential inliers, capped parallax, and anchor-gap penalty. The penalty is the same for the two candidates for a given pair. Quality is a heuristic, not guaranteed to correlate with ground-truth pose error.
- When both fail, report the stronger rejected candidate but keep `tracking_success=0`.
- Add diagnostics: `descriptor_status`, `flow_status`, `descriptor_valid`, `flow_valid`, `both_valid`, per-source geometric inliers and parallax, and pose disagreement (rotation).
- Preserve output naming and all original V3 trajectory columns; additional columns are appended.
- No reference poses are used for frame-by-frame estimation, and no rejected update is converted into a successful translation.

## 600-frame pilot example (PowerShell)

```powershell
$root = "$env:USERPROFILE\Downloads\archaeo_sequence_6_raw_data"
$py = "$env:USERPROFILE\aqualoc-venv\Scripts\python.exe"
$engine = "$root\aqualoc_vo_v3_pilot\slam_engine_sequence6_v3.py"
& $py $engine --root $root --max-frames 600 --algorithms ORB --preprocessings clahe --max-features 1500 --flow-points 1800 --flow-fb-px 1.5 --ratio 0.75 --ransac-px 1.5 --min-essential-inliers 15 --min-cheirality-inliers 10 --min-cheirality-fraction 0.06 --min-parallax-deg 0.1 --max-anchor-gap 20 --step-scale gap --source-policy quality --out "$root\paper_research_draft\vo_pilot_dual_source"
if ($LASTEXITCODE -ne 0) { throw "VO run failed" }
```

Use `--source-policy descriptor_first` on a **different** `--out` folder for an ablation.

## Important interpretation

Your prior replacement already checked both sources and preferred accepted poses. Thus the new script chiefly changes the quality-selection heuristic, adds diagnostic fields, and enables an explicit descriptor-first ablation. Do not assume a huge improvement or guarantee an 80% acceptance rate.

`tracking_success=1` means only that the estimated relative pose passed the chosen geometric checks. These poses remain arbitrary-scale monocular VO, not metric full SLAM. Evaluate trajectory accuracy on independent reference frames and include coverage, held-out accuracy, and failure rate. `anchor_gap` in the output is zero on successfully reanchored frames; use `estimated_anchor_gap` for the gap of the attempted transition.
