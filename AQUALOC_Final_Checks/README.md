# Final submission audit evidence collection

Download/unzip this folder directly under `C:\Users\Raaghav\Downloads\archaeo_sequence_6_raw_data` and run in PowerShell:

```powershell
$root = 'C:\Users\Raaghav\Downloads\archaeo_sequence_6_raw_data'
powershell -NoProfile -ExecutionPolicy Bypass -File "$root\AQUALOC_Final_Checks\COLLECT_FINAL_EVIDENCE.ps1"
```

Result: `paper_research_draft\AQUALOC_Final_Submission_Evidence.zip`.

**Read-only to original data:** original script/CSV/YAML files are copied, not overwritten. The output directory and ZIP are replaced on repeat runs.

**Caution:** the evidence package includes locally created trajectory CSV and calibration/script copies. Review before sharing. Source frame PNGs are not copied.

The included temporal sensitivity bootstrap is a **paired dyadic endpoint-block sensitivity analysis** using the 14,365 unique keyframe pairs per detector, not an independently validated coverage-correct CI estimator. It does not resolve within-sequence tuning.

**Not automatically checkable:** external COLMAP export provenance, true pose quaternion semantic documentation, physical-scale preservation, complete independent-sequence experiments, human-approved authorship/funding/conflict declarations.
