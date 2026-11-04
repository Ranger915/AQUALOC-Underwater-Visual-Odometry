# AQUALOC Sequence 06 — Submission Preparation v2.2

**Contents:** Revised Markdown paper (without inserted figures), read-only frozen-source verifier, full-sequence pose/RPE sensitivity audit, venue and author declaration placeholders, and a prespecified independent-sequence validation protocol.

**No experiments have been rerun using the user's Windows source CSVs in this package.** That dataset is not available inside the file-authoring environment. A manuscript editing pass cannot pass the source-recompute, local reference semantics, or independent-sequence validations by itself.

## Installation
Unzip the package to:
`C:\Users\Raaghav\Downloads\archaeo_sequence_6_raw_data\AQUALOC_Submission_Preparation_V2_2`

## PowerShell run (no need to modify existing experimental results)
```powershell
$root = 'C:\Users\Raaghav\Downloads\archaeo_sequence_6_raw_data'
$py = "$env:USERPROFILE\aqualoc-venv\Scripts\python.exe"
$pkg = Join-Path $root 'AQUALOC_Submission_Preparation_V2_2'
& $py "$pkg\scripts\audit_frozen_results.py" --root $root --hash-images
# Only continue once every discrepancy is resolved; a failure exit code means the audit did not pass.
& $py "$pkg\scripts\reference_pose_rpe_audit.py" --root $root --rpe-gap 20
notepad "$root\paper_research_draft\submission_audit\audit_report.md"
```

To run AP temporal-block sensitivity, locate the previously downloaded `paired_ap_bootstrap.py` and run `--root $root --reps 1000 --blocks 1 5 10 20 --seed 20261104` after preserving its existing output CSV; see §12 of the manuscript.

## Status summary
**Prepared in manuscript:** primary-paper figure slots (not inserted), correct VO labeling, modern related-work comparison, venue-aware required section templates and scientific limitations, audit scripts, reference-format facts from the official AQUALOC paper.

**Still objectively open:** actual dataset re-audit / checksum generation on local user files; confirming whether the particular selected local COLMAP export's `tx/ty/tz` stores `Twc` or `Tcw`, and whether it has the official pressure-scaled coordinate units; independent sequence validation; alternative bootstrap results; author-approved affiliations/funding/conflicts/data archive; actual submission in the official publisher template; and image insertion.

Do not claim independent ground-truth ATE or full SLAM. Do not label benchmark threshold-response heatmaps as physical environment cost maps.
