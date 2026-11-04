# AQUALOC Sequence 06 — non-destructive Adaptive CLAHE extension

This package contains a **copy** of the original frontend, a reusable adaptive module, and an adaptive-only benchmark runner. Existing RAW and fixed-CLAHE output files are not touched.

## Important research caveat
The **v1** rule is specified from the conversation: RAW when global std >=35 AND mean local std >=15; strong CLAHE when global std <25 AND local std <10; moderate otherwise.
The **v2** rule is **inferred**, not confirmed from your actual v2 code: RAW when mean local std >=10.5, otherwise strong CLAHE when both severe conditions hold, moderate CLAHE otherwise. Confirm this against your actual v2 script before treating its output as a reproduction. The reported v1 distribution (808 RAW / 2577 moderate / 0 strong) is **not** assumed to be the v2 distribution.

Threshold 10.5 was selected after inspecting Sequence 06 contrast statistics; therefore v2 is sequence-specific and should be labelled exploratory. Do not claim transferability or an F1 improvement without new evaluation.

## Windows PowerShell

Unzip the package into any folder, e.g. `C:\Users\Raaghav\Downloads\aqualoc_adaptive_package`.

```powershell
$py = "$env:USERPROFILE\aqualoc-venv\Scripts\python.exe"
$script = "$env:USERPROFILE\Downloads\aqualoc_adaptive_package\aqualoc_adaptive_benchmark.py"
$images = "$env:USERPROFILE\Downloads\archaeo_sequence_6_raw_data\images_sequence_6"
& $py $script --image-dir $images --version v1 --max-frames 30
& $py $script --image-dir $images --version v2 --max-frames 30
```

**For full runs, use different output directories** from smoke tests, because the runner refuses to overwrite prior outputs:

```powershell
& $py $script --image-dir $images --version v1 --output-dir "$env:USERPROFILE\Downloads\archaeo_sequence_6_raw_data\sequence6_adaptive_v1_full"
& $py $script --image-dir $images --version v2 --output-dir "$env:USERPROFILE\Downloads\archaeo_sequence_6_raw_data\sequence6_adaptive_v2_full"
```

## Outputs
- `adaptive_frame_decisions.csv`: one row per frame, contrast scores, RAW/moderate/strong decision, CLAHE clip limit.
- `adaptive_feature_pair_metrics.csv`: same pair-level metric column names as original frontend; 3 detector configurations only.
- `adaptive_metadata.json`: parameters, frame and pair counts, original script SHA256, tool versions.

The full 3385-frame sequence with strides 1, 5, 20 should generate 10,129 pairs per detector (30,387 adaptive-only records), assuming no missing frames and no detector failure.

## Scope
This script **does not** calculate mutual matching, COLMAP reference-geometry Sampson errors, precision/recall/F1, AP, ATE, or trajectory estimates. For those, reuse `adaptive_clahe.apply_adaptive` in the existing calibrated and mutual matching pipelines. Keep their reference pairing and thresholds fixed. Compare v1 and v2 separately, and report segment-level or block-bootstrap uncertainty because frames are temporally dependent.

## Methodological warning
An image can be RAW in one frame and CLAHE in the next; measure RAW–CLAHE mixed-pair performance separately. A 10.5 local contrast threshold was chosen based on the observed Sequence 06 histogram, so avoid presenting v2 as an independently pre-registered improvement.
