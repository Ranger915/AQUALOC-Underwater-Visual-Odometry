$ErrorActionPreference = 'Stop'
$root = 'C:\Users\Raaghav\Downloads\archaeo_sequence_6_raw_data'
$py = "$env:USERPROFILE\aqualoc-venv\Scripts\python.exe"
$pkg = Join-Path $root 'AQUALOC_Submission_Preparation_V2_2'
if (-not (Test-Path $py)) { throw 'aqualoc-venv Python not found' }
if (-not (Test-Path $pkg)) { throw "Package not found: $pkg" }
& $py "$pkg\scripts\audit_frozen_results.py" --root $root --hash-images
if ($LASTEXITCODE -ne 0) { Write-Warning 'Audit contains FAIL results. Review report; do not clear this gate.' }
& $py "$pkg\scripts\reference_pose_rpe_audit.py" --root $root --rpe-gap 20
if ($LASTEXITCODE -ne 0) { throw 'Pose convention sensitivity audit failed' }
Write-Host "Review $root\paper_research_draft\submission_audit" -ForegroundColor Cyan
