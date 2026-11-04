# Full-sequence AQUALOC V3 trajectory evaluation

This program evaluates `paper_research_draft/vo_full_sequence_v3/vo_sequence6_orb_clahe_v3.csv` against `colmap_traj_sequence_6.txt` using **fresh accepted VO poses only**, with a chronological 60/40 train/test split. It estimates one Sim(3) similarity transform from the earlier accepted reference-indexed poses and scores the later held-out accepted poses. All metrics are in *COLMAP reconstruction units*, not verified metres.

## Run (Windows PowerShell)

Extract ZIP within `C:\Users\Raaghav\Downloads\archaeo_sequence_6_raw_data` and run:

```powershell
$root = 'C:\Users\Raaghav\Downloads\archaeo_sequence_6_raw_data'
$py = "$env:USERPROFILE\aqualoc-venv\Scripts\python.exe"
& $py "$root\aqualoc_full_sequence_eval\evaluate_full_v3.py" --root $root --reference-convention twc --min-train 10 --min-test 5
```

The `twc` flag **asserts**, but does not verify, the reference pose convention. Dataset documentation or source code defining COLMAP trajectory export is needed to validate whether the first 3 position fields are world-frame camera centers (`twc`) or world-to-camera translations (`tcw`). Do not choose based on whichever gives a lower metric.

Results: `paper_research_draft/full_sequence_trajectory_evaluation/{summary.json,summary_flat.csv,fresh_pose_errors.csv,01..08_*.png}`.

`--max-staleness 5` additionally reports staleness-qualified errors as a *separate* diagnostic using the **same alignment fit**; this must not be mixed with fresh tracking accuracy. Carried-forward poses are not included in the primary ATE-like statistics.

**Warning:** This is *COLMAP-relative, Sim(3)-aligned translational consistency*, not independent ground-truth metric ATE; the estimator has arbitrary monocular scale. The earlier 60% of jointly fresh reference-indexed poses are used to calibrate a global alignment, and later 40% form a *chronological holdout within the same dataset*. Model selection has already used Sequence 06; do not label this an independent dataset test. The resulting trajectory error reflects accumulated failures as well as accepted poses, and the reference frame convention remains to be verified.
