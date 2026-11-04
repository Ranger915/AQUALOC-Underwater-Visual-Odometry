$ErrorActionPreference = 'Stop'
$root = 'C:\Users\Raaghav\Downloads\archaeo_sequence_6_raw_data'
$dest = Join-Path $root 'paper_research_draft\submission_final_evidence'
$py = Join-Path $env:USERPROFILE 'aqualoc-venv\Scripts\python.exe'
New-Item -ItemType Directory -Force -Path $dest | Out-Null
if (-not (Test-Path $py)) { throw "Missing Python $py" }
$rootPath = (Resolve-Path $root).Path

# 1. precise calibration and trajectory export provenance
$filesToCollect = @(
  'colmap_traj_sequence_6.txt',
  'new_archaeo_colmap_traj_sequence_06.txt',
  'aqualoc_vo_v3_pilot\slam_engine_sequence6_v3.py',
  'paper_research_draft\full_sequence_trajectory_evaluation\summary_flat.csv',
  'paper_research_draft\vo_full_sequence_v3\vo_sequence6_orb_clahe_v3.csv',
  'sequence6_paper_validation\adaptive_v2_mutual_ap_paired_bootstrap.csv',
  'sequence6_paper_validation\adaptive_reference_f1_bootstrap.csv'
)
$orig = Join-Path $dest 'selected_inputs'
New-Item -ItemType Directory -Force -Path $orig | Out-Null
$inventory = foreach ($relative in $filesToCollect) {
    $path = Join-Path $root $relative
    if (Test-Path $path -PathType Leaf) {
      $target = Join-Path $orig ($relative -replace '[\\/]', '__')
      Copy-Item -LiteralPath $path -Destination $target -Force
      [pscustomobject]@{Input=$relative;Exists=$true;SHA256=(Get-FileHash $path -Algorithm SHA256).Hash;Bytes=(Get-Item $path).Length}
    } else {[pscustomobject]@{Input=$relative;Exists=$false;SHA256='';Bytes=0}}
}
$inventory | Export-Csv (Join-Path $dest 'important_input_inventory.csv') -NoTypeInformation

# Calibration YAMLs: copy all actual matches, not just the first ambiguous match
$cal = Join-Path $dest 'calibration'
New-Item -ItemType Directory -Force -Path $cal | Out-Null
Get-ChildItem -LiteralPath $root -Recurse -File -ErrorAction SilentlyContinue |
 Where-Object { $_.Name -in @('archaeo_camera_calib.yaml','archaeo_imu_camera_calib.yaml','archaeo_imu_noises.yaml') } |
 ForEach-Object {
   $safe=($_.FullName.Substring($rootPath.Length).TrimStart('\') -replace '[\\/]', '__')
   Copy-Item -LiteralPath $_.FullName -Destination (Join-Path $cal $safe) -Force
 }

# Full code/runner file inventory; retain hashes without zipping entire image dataset
$codeFiles = Get-ChildItem -LiteralPath $root -Recurse -File -ErrorAction SilentlyContinue |
 Where-Object { $_.Extension -in @('.py','.ps1','.yaml','.yml','.toml','.ini','.json','.md') -and $_.FullName -notmatch '[\\/]submission_final_evidence[\\/]' }
$codeFiles | ForEach-Object {
  [pscustomobject]@{Relative=$_.FullName.Substring($rootPath.Length).TrimStart('\');SHA256=(Get-FileHash $_.FullName -Algorithm SHA256).Hash;Bytes=$_.Length}
} | Export-Csv (Join-Path $dest 'code_file_inventory.csv') -NoTypeInformation

# Dataset inventory (other sequences) and key script usage
Get-ChildItem -LiteralPath (Join-Path $env:USERPROFILE 'Downloads') -Directory -ErrorAction SilentlyContinue |
 Where-Object { $_.Name -match '(?i)aqualoc|archaeo|sequence' } |
 Select-Object Name,FullName |
 Export-Csv (Join-Path $dest 'available_sequence_folders.csv') -NoTypeInformation
& $py (Join-Path $root 'aqualoc_vo_v3_pilot\slam_engine_sequence6_v3.py') --help *>&1 |
 Out-File (Join-Path $dest 'v3_engine_help.txt')
& $py -m pip freeze | Out-File (Join-Path $dest 'pip_freeze.txt')
& $py -c 'import cv2; print(cv2.getBuildInformation())' | Out-File (Join-Path $dest 'opencv_build.txt')
Get-CimInstance Win32_Processor | Select-Object Name,NumberOfCores,NumberOfLogicalProcessors | Export-Csv (Join-Path $dest 'cpu.csv') -NoTypeInformation
Get-CimInstance Win32_VideoController | Select-Object Name,DriverVersion | Export-Csv (Join-Path $dest 'gpu.csv') -NoTypeInformation
Get-CimInstance Win32_OperatingSystem | Select-Object Caption,Version,BuildNumber | Export-Csv (Join-Path $dest 'windows.csv') -NoTypeInformation

# small, unambiguous data preview required for pose convention verification
Get-Content (Join-Path $root 'colmap_traj_sequence_6.txt') -TotalCount 12 | Out-File (Join-Path $dest 'reference_first_12_lines.txt')
Get-Content (Join-Path $root 'new_archaeo_colmap_traj_sequence_06.txt') -TotalCount 12 | Out-File (Join-Path $dest 'reference_alternate_first_12_lines.txt')

# Preserve uploaded previous audits if they exist
$previous = Join-Path $root 'paper_research_draft\submission_audit'
if (Test-Path $previous) {
  $prevOut=Join-Path $dest 'prior_audit'
  New-Item -ItemType Directory -Force -Path $prevOut | Out-Null
  Get-ChildItem -LiteralPath $previous -File -ErrorAction SilentlyContinue |
    Where-Object { $_.Name -notin @('reproducibility_manifest.json') } |
    Copy-Item -Destination $prevOut -Force
}

# bootstrap sensitivity: script is expected alongside this runner
$boot = Join-Path $PSScriptRoot 'bootstrap_sensitivity.py'
if (-not (Test-Path $boot)) { throw "Missing $boot" }
& $py $boot --root $root --out (Join-Path $dest 'bootstrap_sensitivity') --reps 400 --seed 20261105 --blocks '1,5,10,20'
if ($LASTEXITCODE -ne 0) { throw 'Bootstrap sensitivity failed' }

# Package; last operation excludes itself (is outside directory)
$zip=Join-Path $root 'paper_research_draft\AQUALOC_Final_Submission_Evidence.zip'
if (Test-Path $zip) { Remove-Item $zip -Force }
Compress-Archive -Path (Join-Path $dest '*') -DestinationPath $zip -CompressionLevel Optimal
Write-Host "`n[DONE] Evidence archive: $zip" -ForegroundColor Green
Get-Item $zip | Select-Object FullName,Length,LastWriteTime | Format-List
