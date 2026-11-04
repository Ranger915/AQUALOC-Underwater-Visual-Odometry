# Prespecified independent-sequence validation protocol — NOT YET EXECUTED

## Goal
Test whether the fixed preprocessing and matching policy selected on AQUALOC Archaeological Site Sequence 06 generalizes to other AQUALOC sequences **without further tuning**. The original study is exploratory; independent validation requires access to images, calibration, and COLMAP references for a disjoint sequence.

## Freeze before any other sequence is inspected
1. Hash and archive the exact finalized source of `slam_engine_sequence6_v3.py`, the adaptive preprocessing implementation, matching scripts and configuration JSON/YAML.
2. Record frozen policy: ORB, fixed CLAHE clip=2.0 and 8×8 tiles, feature budget 1500, Lowe threshold 0.75, flow budget 1800, forward–backward 1.5 px, 1.5-px essential RANSAC, 15 essential/10 cheirality minimum, minimum cheirality fraction 0.06, min parallax 0.1 degree, max anchor gap 20, geometric quality selection, and arbitrary gap-based translation scaling.
3. Declare the output metrics before the new run: per-sequence accepted fraction, reference-fresh coverage, number of jointly evaluable reference poses, chronologically held-out Sim(3) positional RMSE/median/p95 with unchanged test fraction; relative pose error only if conventions confirmed.
4. Record the candidate sequence names and reason(s) for selection **before** computing any of these outcomes. Use every sequence in the prespecified set including failed or untrackable sequences. Prefer at least two independent sequences with varied texture/depth and supply their own camera calibration and frame sampling.

## Dataset caveat
The packaged V3 engine uses Sequence 06 assumptions and an internal camera model. **Do not run it unmodified on a different sequence** unless camera calibration, image count/index convention, timestamps, valid filenames and evaluation association are all made sequence-specific. Such changes must be purely data-interface adaptations, not outcome-driven motion-threshold changes. Cross-sequence GT labels, loop matrices and spatial scale have to be rebuilt/loaded for the new sequence; never reuse Sequence 06 pair manifest or score labels.

## No data leakage
Do not adjust cheirality, match source, CLAHE thresholds, alignment selection, normalization, frame filtering, or loss handling based on independent-sequence results. Train/alignment split must be chosen by chronological frame IDs. Sim(3) calibration must use only the designated alignment poses, never held-out test poses. Keep the entire rejected-transitions log and use identical frame eligibility criteria across configurations.

## Failure and reporting template
| Sequence | Frozen config SHA-256 | Images processed | Accepted/attempted | Fresh reference poses/available | Alignment n | Holdout n | Sim(3)-relative RMSE and units | Failure reason | Source file hashes |
|---|---|---:|---:|---:|---:|---:|---|---|---|
| **Not yet run** | — | — | — | — | — | — | — | Awaiting independent dataset and verified adaptation | — |

A complete negative/failure outcome must remain visible in the paper or supplements. Under no circumstances mark this submission requirement checked based on Sequence 06 results alone.
