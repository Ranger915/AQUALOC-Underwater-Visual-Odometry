# From Feature Correspondence to Trajectory Reliability in Underwater Visual Odometry: A Calibrated Study of Contrast Enhancement on AQUALOC Sequence 06

**Manuscript version:** 2.2 — submission-preparation draft with verified literature, an audit package, reproducibility declarations, and an explicit evidence-gated checklist
**Date:** 5 November 2026
**Author:** B. Raaghavan Mookiah
**Study:** AQUALOC archaeological-site Sequence 06
**Manuscript category:** Single-sequence empirical benchmark, negative-result study, and exploratory visual-odometry ablations
**Provisional target venue:** *Sensors* (MDPI), original research article; final choice, APC, author consent, affiliation, and template conversion remain subject to the author.
**Submission status:** **NOT CLEARED**. Literature, manuscript labeling, audit tooling and a proposed venue are prepared. Numerical re-audit on the researcher's local CSVs, reference-frame export semantics, independent-sequence replication, and author declarations remain open. Figures have intentionally **not** been inserted.

> **Evidence convention.** `[C]` = completed experiment with numerical values transcribed from project-run output; `[E]` = exploratory experiment on Sequence 06, including parameter decisions made after examining the sequence; `[U]` = unverified assumption or required check; `[P]` = proposed, not measured. This manuscript preserves reported experimental values but does not claim that the authoring environment has independently rerun the original 3,385-image dataset or verified every source CSV. Figures are **not embedded as if they existed**: §13 provides exactly ten suggested figure files, expected folder paths, captions, and availability notes.

---

## Abstract

Reliable feature correspondence is necessary for underwater visual localization, but neither plentiful descriptor matches nor success on pairwise epipolar geometry guarantees dependable camera-motion estimation. We investigate this discrepancy using AQUALOC Archaeological Site Sequence 06 (3,385 images, approximately 20 Hz), comparing ORB, AKAZE-MLDB, and SIFT under unprocessed grayscale, fixed contrast-limited adaptive histogram equalization (CLAHE), and an image-wise contrast-adaptive preprocessing rule. The study begins with 60,774 sequential pairwise feature-benchmark records, extends to calibrated reference matching over 170 COLMAP keyframes and 14,365 unordered pairs, and tests mutual descriptor agreement, local/nonlocal overlap detection, matching density, preprocessing latency, and paired block-bootstrap uncertainty. At the 2-pixel, eight-inlier operating point, fixed CLAHE outperforms adaptive preprocessing on overall mutual-matching F1 for ORB (0.8416 versus 0.8376), AKAZE-MLDB (0.5478 versus 0.4964), and SIFT (0.8797 versus 0.8673). Relative to fixed CLAHE, adaptive count-based average-precision differences are −0.0059 for ORB (95% bootstrap interval [−0.0164, 0.0004]), −0.0195 for AKAZE-MLDB ([−0.0497, −0.0016]), and −0.0107 for SIFT ([−0.0236, −0.0008]). Adaptive processing costs 5.655 ms/image versus 0.363 ms/image for fixed CLAHE. A downstream, explicitly **monocular visual-odometry** extension uses descriptor and optical-flow correspondences, adaptive keyframe anchoring, and cheirality-based pose validation. Its frozen full-sequence ORB–CLAHE configuration accepts 1,188 of 3,384 frame transitions (35.11%); only 60 of 170 COLMAP-reference keyframes have fresh accepted updates. A chronological, 36-pose Sim(3) fit followed by 24 held-out reference poses produces a translational RMSE of 4.089 **COLMAP reconstruction units** under an assumed reference convention. Controlled 600-frame ablations further show that relaxing geometric acceptance can increase accepted updates while worsening held-out trajectory error. We conclude that underwater feature evaluation should distinguish correspondence redundancy, geometrically recoverable motion, tracking coverage, accumulated drift, and computational cost. Results are sequence-specific and reconstruction-referenced; metric ground-truth SLAM accuracy is not claimed.

**Keywords:** underwater visual odometry; AQUALOC; feature matching; CLAHE; adaptive contrast; ORB; AKAZE; SIFT; optical flow; essential matrix; cheirality; Sim(3); statistical validation.

## 1. Introduction

Underwater image formation is affected by attenuation, nonuniform artificial illumination, limited texture, and changing visual appearance. Such effects can alter feature detection and description and challenge monocular navigation [1]. Local contrast enhancement is attractive because it often reveals otherwise weak image structures. Yet preprocessing applied independently to successive images may also disturb descriptor consistency and alter the geometric support available for essential-matrix decomposition. Furthermore, the successful fitting of a pairwise fundamental matrix should not be confused with a stable sequence-wide camera trajectory.

This work is motivated by a practical discrepancy observed during a complete experimental pipeline. Fixed CLAHE produced strong reference-based correspondence metrics, whereas an intuitively appealing frame-wise adaptive rule was generally inferior. Subsequently, a monocular VO engine identified large numbers of tentative matches but accepted relatively few camera-motion estimates; optimizing the acceptance gate increased pose counts without necessarily improving trajectory agreement. The study thus considers the imaging, correspondence, two-view geometry, and trajectory stages as separate objects of measurement rather than as interchangeable proxies for localization accuracy.

**Research questions.** **RQ1:** How do original grayscale images and fixed CLAHE affect correspondence density and calibrated reference-pair discrimination for ORB, AKAZE-MLDB, and SIFT? **RQ2:** Does a frame-wise contrast-adaptive rule improve these outcomes, including uncertainty and computational cost? **RQ3:** How far does successful matching carry through essential-matrix recovery, cheirality filtering, and integrated monocular visual odometry? **RQ4:** Can greater pose acceptance be mistaken for better held-out trajectory consistency?

**Contributions.** First, a full-sequence, three-detector, multi-stride frontend benchmark is paired with a reference-geometric evaluation over all unordered pairs of 170 COLMAP keyframes. Second, adaptive preprocessing is evaluated against a same-code fixed-CLAHE control with bootstrap uncertainty, latency and mixed-preprocessing-pair audits. Third, a sequence-level VO diagnostic separates descriptor matching, optical-flow tracking, essential-matrix support, cheirality and reference-pose coverage. Fourth, controlled pilot ablations provide a concrete counterexample to equating accepted-pose count with localization quality. We make **no claim of a novel full SLAM system** or universal optimality of fixed CLAHE.

## 2. Related Work

AQUALOC was introduced as an underwater monocular–inertial–pressure dataset with trajectories derived offline using structure from motion, precisely because independent underwater ground truth is difficult to acquire [1]. SIFT offers scale-space keypoints and continuous-valued descriptors [2], whereas ORB uses computationally efficient binary descriptions [3], and AKAZE constructs features in nonlinear scale spaces [4]. These complementary detectors offer a meaningful comparison across feature response and descriptor families. CLAHE restricts local histogram amplification while improving localized contrast [5]. COLMAP provides a reference SfM framework and geometric reconstructions, but offline trajectories should not be conflated with physically measured truth [6]. Lucas–Kanade optical flow supplies a complementary short-range image-tracking mechanism [7]. Common trajectory benchmark conventions distinguish global alignment error from relative motion error and require explicit synchronization, scale and ground-truth definitions [8].

Existing techniques explain why preprocessing or optical flow might affect correspondence performance, but they do not establish that **this particular contrast-switching rule** improves geometry on **this particular underwater sequence**. The present study makes that question empirical. It also foregrounds failure/coverage statistics that may be hidden when image enhancement is evaluated solely through feature counts or in-sample pairwise geometry. A comprehensive claim of novelty relative to all recent underwater SLAM literature is **not** made. A dated contemporary study comparison is provided below; we do not claim superiority over these systems because their sensors, inputs, scenes and outcome metrics differ.

### 2.1. Contemporary related-work comparison (reviewed 5 November 2026)

| Work and year | Method category | Input and scope | Reported evaluation focus | Relationship to this study |
|---|---|---|---|---|
| Leonardi et al., UVS (2023) [9] | Underwater monocular visual SLAM | Camera-only, mapping, initialization and loop closures; includes AQUALOC evaluations | SLAM trajectory and robustness | A **full SLAM comparator**, unlike our VO; no same-setting head-to-head test |
| Xin et al., ULL-SLAM (2023) [10] | Low-light enhancement for SLAM frontends | Image enhancement for underwater front-end | Enhanced feature matching / SLAM implications | Motivation for preprocessing comparisons; no direct shared test protocol |
| Drupt et al., MAM³SLAM (2024) [11] | Multi-agent monocular visual SLAM | Multiple ROVs and centralized mapping | Multi-agent localization and mapping | Different sensing/task configuration; not directly comparable |
| Xiang et al. (2024) [12] | Image enhancement plus sonar fusion | Visual/sonar SLAM | Navigation robustness with sensor fusion | Cannot be compared numerically with monocular-only VO |
| Chen et al. (2025) [13] | Underwater SLAM with adaptive enhancement | Adaptive low-light/turbid recognition and enhancement | End-to-end underwater SLAM | A distinct adaptive learned/fusion method; our simple contrast rule is **not** a counterexample to their claims |
| Summers et al. (2025) [14] | Underwater enhancement evaluation | Feature matching stability and SLAM implications | Local matching stability, furthest matchable frame | Particularly close methodological precedent; reinforces downstream evaluation |
| Li et al., Ucolor (2021) [15] | Underwater image enhancement only | Medium-transmission-guided multi-color enhancement | Image restoration | Not a localization/VO study; no direct ATE comparison |
| An et al., DFSVO (2026) [16] | Underwater stereo visual odometry | Stereo camera, detection-free tracking | Localization and mapping performance | Different stereo modality; do not compare raw ATE values |
| **This work** | Fixed/adaptive preprocessing + monocular VO reliability audit | AQUALOC Sequence 06 monochrome; 3 feature families; no sensor fusion | Pairwise F1/AP, block bootstrap, acceptance, reconstruction-relative held-out drift | Negative result and multi-stage diagnostic; **single-sequence exploratory VO** |

The row descriptions are based on the cited papers' public abstracts and publisher records, not a reproduced run of their software. The closest recent enhancement-versus-matching paper [14] must be discussed prominently in novelty assessment and during any journal submission.

## 3. Data, Calibration, and Reference Provenance

### 3.1. Sequence and timing audit [C]

AQUALOC Archaeological Site Sequence 06 includes **3,385** monochrome images of **968 × 608** pixels. The audited image timestamp table begins at **1542883311780022336 ns** and ends at **1542883480951782752 ns**, for a duration of **169.171760416 s**. Timestamps increase monotonically, with zero duplicates. The median interval is **49.996 ms**, the observed range **18.993–76.665 ms**, and the mean frequency approximately **20.003 Hz**. The corresponding raw archive is `archaeo_sequence_6_raw_data`, with source images in `raw_data/images_sequence_6` and timestamps in `img_sequence_6.csv` (verify the actual on-disk location before packaging a reproducibility release).

The raw archive also contains IMU and magnetometer samples and a depth-associated CSV. Three calibration files were downloaded: `archaeo_camera_calib.yaml`, `archaeo_imu_camera_calib.yaml`, and `archaeo_imu_noises.yaml`. Although these support future visual–inertial analysis, the present trajectory experiments do **not** integrate the IMU; a depth CSV alone does **not** establish the availability of registered depth images sufficient for metric RGB-D PnP.

### 3.2. Camera calibration [C]

| Parameter | Value |
|---|---:|
| Camera model | Pinhole |
| Distortion model | Radial–tangential |
| Resolution | 968 × 608 |
| \(f_x\) | 543.3327734182214 |
| \(f_y\) | 542.398772982566 |
| \(c_x\) | 489.02536042247897 |
| \(c_y\) | 305.38727712002805 |
| \(k_1\) | −0.1255945656257394 |
| \(k_2\) | 0.053221287232781606 |
| \(p_1\) | 0.0000994070021080493 |
| \(p_2\) | 0.00009550660927242349 |

The calibrated intrinsic matrix is

\[
K=\begin{bmatrix}543.3327734&0&489.0253604\\0&542.3987730&305.3872771\\0&0&1\end{bmatrix}.
\]

These values and radial–tangential distortion are used when interpreting calibrated image correspondences and the later Sequence 06 VO experiments. The underlying camera–IMU extrinsics and noise-model YAMLs should be included in the supplementary materials even though their measurements were not fused in the reported monocular VO runs.

### 3.3. COLMAP-derived products and overlap labels [C, U]

Two eight-column pose exports, `colmap_traj_sequence_6.txt` and `new_archaeo_colmap_traj_sequence_06.txt`, contain a first-column frame ID and three translation-like entries followed by four quaternion entries. Each has **170** reference poses at frame indices **0, 20, …, 3380**. A previous similarity-alignment audit found **166 common frame IDs**, estimated scale **0.464995233081361**, and positional residual RMSE **3.97 × 10⁻¹²** after Sim(3) alignment. This establishes equivalence under similarity alignment of the compared exported positions, **not** independent reconstruction accuracy or metric calibration.

The reference overlap matrix `colmap_detected_loop_sequence_6.txt` is **170 × 170**, binary, symmetric and has a unit diagonal. Over the **14,365 unordered off-diagonal keyframe pairs**, **1,143** are positive (approximately **7.96%**) and **13,222** negative. This is an **offline reference overlap relation**; it is not a loop detector implemented by the new VO engine. The reported reference baseline distribution has median **2.289** in reconstruction units (range **0.00867–8.406**) and relative rotation median **66.51°**, with maximum near **179.99°**. Do not add “m” to these baseline measurements until their metric scale is independently established.

**Verified from the dataset authors:** Ferrera et al. [1] state that the public COLMAP-derived trajectory baseline was **scaled using pressure-sensor measurements** to provide metric positions, while explicitly cautioning that it is not perfect ground truth. Their documented text export orders fields as `frame, tx, ty, tz, qx, qy, qz, qw` and uses every 20th frame in archaeological-site sequences. This documents the **dataset authors’ intended released baseline**, not the exact transformation used for the two *local* Sequence 06 exports: those local files differ by Sim(3) scale ≈0.465, and which export retains the pressure-based metric scaling is not established. Separately, COLMAP's native `images.txt` format uses a world-to-camera pose with quaternion order `qw,qx,qy,qz`; the AQUALOC eight-field text is a converted/exported **different column order**. Therefore native COLMAP conventions cannot be blindly copied into the AQUALOC text reader. The unit and coordinate convention of the specific selected local file remain **unverified**; the existing 4.089 result stays labeled in export/reconstruction units. The AQUALOC paper's camera–IMU extrinsic equation is `T_world_camera = T_world_imu · T_imu_camera`; because our optical VO is already camera-referenced, no IMU extrinsic should be applied unless comparing with an IMU-referenced pose.

**Outstanding convention check [U].** Our trajectory evaluators have used `--reference-convention twc`, assuming the stored translation is a camera center in world coordinates. The eight columns alone cannot prove that. If the export instead stores world-to-camera translation, camera centers must be computed as \(\mathbf C_w=-R_{cw}^{\top}\mathbf t_{cw}\), and the numerical ATE must be rerun. The final article cannot legitimately describe the current trajectory RMSE as verified ground-truth accuracy or as metres before this is resolved.

### 3.4. Reference-pair partitioning [C]

From 170 poses there are \(\binom{170}{2}=14,365\) unordered keyframe pairs, the actual pair manifest size. The **local** partition, image-frame separation ≤100, contains **835** pairs, **660** reference positives. The **nonlocal** partition, separation ≥200, contains **12,880** pairs, **246** reference positives. The remaining intermediate-gap partition is neither local nor nonlocal, so the two reported partitions do not exhaust all pairs. Local and nonlocal experiments must always report their distinct positive prevalences.

## 4. Experimental Methods and Chronological Development

### 4.1. Initial full-sequence uncalibrated frontend [C]

The earliest experiment explored ORB, AKAZE-MLDB and SIFT on raw grayscale versus fixed CLAHE images, applying a cap of **1,500 keypoints/image**. Fixed CLAHE used `clipLimit=2.0` and `tileGridSize=(8,8)`. Benchmarked descriptor matching used the Lowe ratio test (nominal threshold **0.75**), Hamming distance for binary descriptors and Euclidean distance for SIFT. Fitted fundamental-matrix RANSAC used a nominal **1.5-pixel** threshold and confidence **0.99**. Image pairs were separated by **1**, **5**, or **20** frame indices: 3,384 + 3,380 + 3,365 = **10,129** pairs for each detector/preprocessing combination, giving **60,774** rows for the six original configurations. Before the full run, an initial 100-image frontend pilot evaluated **274** pairs/configuration, six configurations, **1,644** pair assessments.

Measurements included detected and matched features, RANSAC inliers, inlier fraction, epipolar residual, and occupied grid cells. The full frontend established feature and temporal-gap behavior, but because its fundamental matrix was estimated using the same candidate matches, it could **not** independently establish true pairwise overlap.

The runtime environment must be regenerated by `scripts/audit_frozen_results.py` on the actual laptop. The OpenCV environment required explicit attention: a Microsoft Store Python 3.13 environment initially exposed OpenCV 5.0.0 where `cv2.AKAZE_create` was absent but `cv2.xfeatures2d.AKAZE_create` was present. The dedicated `aqualoc-venv` environment and AKAZE fallback allowed the experiments to complete. Reproducing the exact environment, not merely naming an OpenCV version, is necessary for reproducibility.

### 4.2. Calibrated reference geometry [C]

We constructed a pair manifest over the 170 COLMAP keyframes. Given relative calibrated motion \((R,t)\), the essential matrix is \(E=[t]_\times R\); under compatible camera and pixel-coordinate conventions, the fundamental matrix is \(F=K^{-\top}EK^{-1}\). Using an independent reference rather than re-fitting geometry from a pair's candidate matches reduces circularity in epipolar evaluation, although all references still depend on COLMAP.

The Sampson approximation for putative image correspondence \((x,x')\), using homogeneous coordinates, is

\[
d_S(x,x';F)=\frac{(x'^\top F x)^2}{(Fx)_1^2+(Fx)_2^2+(F^\top x')_1^2+(F^\top x')_2^2}.
\]

Reference-consistent matches were evaluated across equivalent pixel-distance thresholds **1, 2, 3, 5 px**, with reference-inlier-count decision cutoffs **8, 12, 20**. The calibrated non-mutual baseline consists of **86,190** detector–preprocessing–pair records (6 × 14,365). Adaptive v2 later contributed **43,095** additional records (3 × 14,365). Correspondence coordinates were retained in compressed match tables. The geometric and overlap-label calculations depend on the actual pose conventions; this must be audited before final publication.

For a binary overlap prediction \(\hat y_{ij}\) and reference label \(y_{ij}\), precision, recall and F1 are

\[P=\frac{TP}{TP+FP},\qquad R=\frac{TP}{TP+FN},\qquad F_1=\frac{2PR}{P+R}.\]

Average precision (AP) summarizes rank ordering of scored pair predictions across thresholds; the present study reports **count-based AP** and, for mutual matching, separately reports AP from a fraction-based score. Neither score is a percentage of individual inlier correspondences. **Implementation definition:** The saved mutual source code defines 2-pixel Sampson-consistent mutual match count (`mutual_inliers_2px`) and fraction (`mutual_fraction_2px`). A pair is **predicted positive** at the 2-pixel/8-inlier operating point when its count is ≥8. **Count AP** ranks reference-overlap labels using the raw count; **fraction AP** ranks them using the fraction. The binary eight-inlier cutoff does not redefine the ranking AP. The script `audit_frozen_results.py` recalculates both from the pair-level CSV, with ties grouped as in non-interpolated AP, and flags any numerical mismatch. Its success must be confirmed by running it on the local frozen CSVs.

### 4.3. Mutual-matching sensitivity [C]

A separate matching protocol required reciprocal descriptor agreement. It used the same COLMAP-derived reference pairs and geometry rather than a new set of images. The six original mutual configurations contain **86,190** rows, with **43,095** additional adaptive-v2 rows. This test checks whether enhancement comparisons survive a stricter correspondence criterion and whether overlap rankings are stable under a different score definition.

### 4.4. Adaptive contrast decisions: v1 → v2 [C, E]

Initial adaptive v1 used a conditional contrast rule that selected raw images when global gray-level standard deviation was at least **35** and the local contrast statistic at least **15**; other images received CLAHE. Examination of v1 decisions motivated adaptive v2, which selected raw when mean local-region standard deviation was at least **10.5**, otherwise CLAHE with policy-dependent clip strength. The audited v2 decisions on 3,385 frames were **808 raw (23.87%)**, **2,577 moderate CLAHE (76.13%)**, and **zero strong CLAHE**. Consequently, the measured v2 policy is effectively a *raw-versus-moderate-CLAHE switch*, not a three-state enhancement success demonstration.

Because v2 was chosen after inspecting Sequence 06 image statistics, it is **sequence-tuned exploratory preprocessing**, even though the threshold choice preceded inspection of its final matching outcomes. A same-implementation fixed-CLAHE control was subsequently run to avoid attributing code differences to the adaptive policy. In the original sequential frontend, the fixed and adaptive comparisons each comprise **30,387** pair records across all three detectors.

Per detector, the comparison contains **7,705 CLAHE–CLAHE pairs**, **2,372 raw–raw pairs** and **52 mixed-preprocessing pairs** (2 at stride 1; 10 at stride 5; 40 at stride 20). This stratification tests whether image-wise switching destabilizes features within pairs. Of the 16 AKAZE pairs that changed from success to failure under the control's ≥8-inlier criterion, **all 16 were mixed-preprocessing stride-20 pairs**. This is an observed association, not proof that enhancement mismatch alone caused every failure.

### 4.5. Statistical uncertainty and runtime [C]

For the frontend, **1,000 paired block-bootstrap** replicates used approximately **100-frame blocks** to estimate uncertainty in method differences. At the reference-pair level, **1,000 paired bootstrap** replicates used **10-keyframe-anchor blocks** for F1 and AP comparisons in all, local and nonlocal partitions. The AP resampling script checked that the reconstructed AP definition reproduced the original and adaptive summaries before accepting bootstrap output. Intervals below are **nominal percentile 95% intervals**, not corrected for multiplicity.

Blocking recognizes temporal dependence but cannot fully account for pairs sharing both source and target keyframes. In particular, 14,365 reference pairs do not represent 14,365 independent underwater scenes. Findings near zero require block-length and dyadic-dependence sensitivity analyses for strong inferential claims.

A preprocessing-time benchmark repeated evaluation over all 3,385 images **three times** per method. It distinguishes raw pass-through, fixed CLAHE, and adaptive decision/application cost, and excludes feature extraction, matching, pose recovery, visualization and disk I/O. Hardware details must be added from the machine used for each original timing run before submission.

### 4.6. Downstream monocular VO engine evolution [C, E]

This study subsequently used a separately authored `slam_engine.py` supporting ORB/AKAZE, RGB-D PnP *when valid aligned depth is available*, and monocular essential-matrix motion estimation. Here only **monocular** mode was run; the initial engine expected video files, so a Sequence 06 image-sequence runner was implemented. The baseline engine called `findEssentialMat` and `recoverPose` on undistorted matched points, accepted translations only if at least 12 cheirality-valid inliers and a ≥0.20 inlier fraction were obtained, and integrated each accepted relative pose into \(T_{wc}\). Rejected transitions carried forward the last pose. Without an external scale measurement, every accepted monocular translation had an assumed arbitrary step magnitude.

Version 2 introduced camera calibration, adaptive anchor/keyframe association, essential and cheirality diagnostics, parallax checks, and explicit failure statuses. Version 3 allowed unit, square-root-gap and gap-scaled arbitrary steps, then added descriptor/optical-flow candidates, geometric validation for both, source-selection policy and detailed rejection traces. **These are engineering iterations of exploratory VO**, not separate independently validated SLAM systems. There is no established loop-closure optimization, bundle-adjusted map, visual–inertial fusion, or metric-scale recovery. For accepted relative pose \(T_{c_kc_a}\), integration in camera-to-world convention follows \(T_{wc_k}=T_{wc_a}\,T_{c_kc_a}^{-1}\), subject to correct anchor association; pose-error analysis needs to inspect how reanchoring and held coordinates are handled.

For frame-indexed estimated poses and a COLMAP-derived reference, a Sim(3) transform \((s,R,t)\) was fitted on **earlier** joint fresh reference poses:

\[
(s^*,R^*,t^*)=\arg\min_{s>0,R\in SO(3),t}\sum_{i\in\mathcal I_{train}}\|\mathbf p_i^{ref}-(sR\mathbf p_i^{VO}+t)\|^2.
\]

The held-out translational residual is \(e_i=\|\mathbf p_i^{ref}-(s^*R^*\mathbf p_i^{VO}+t^*)\|\), and \(\operatorname{RMSE}=\sqrt{|\mathcal I_{test}|^{-1}\sum_{i\in\mathcal I_{test}}e_i^2}\). A similar alignment on the test poses would not be a held-out evaluation. The result is **COLMAP-relative, Sim(3)-aligned translational trajectory consistency**, not verified metric absolute trajectory error. The `twc` convention was assumed rather than independently checked. Explicit relative pose error (RPE), rotation error and reconstruction scale in metres were **not computed** in the supplied results.

### 4.7. Candidate tracking and failure semantics (implementation-level description)

In V3, ORB descriptors are matched using Lowe-ratio filtering (binary/Hamming). Independently, Shi–Tomasi corners or detector points are tracked between anchor and current frames by pyramidal Lucas–Kanade optical flow, with forward–backward pixel displacement filtering at the configured threshold. Descriptor-derived and flow-derived two-view candidates undergo calibrated essential-matrix RANSAC, `recoverPose` cheirality filtering, minimum inlier counts, cheirality fraction and parallax testing **separately**. For each accepted candidate, V3 records feature and inlier counts and source identity; where two candidates pass, the recorded `quality` policy selects the preferred candidate according to the engine's geometric ranking rule. `descriptor_first` was examined as an ablation, **not** used for the full sequence. The engine uses a maximum keyframe anchor gap of 20 frames; accepted anchor-to-current transforms are integrated into the camera-to-world trajectory. When tracking fails, the previous integrated pose may be carried to the current CSV row; `tracking_success=0` means that this carried pose is **not** a fresh valid motion estimate. The main coverage statistic therefore counts only `tracking_success=1` at the 170 reference IDs. Rotation-only statuses, if present, must not be counted as valid translations. These descriptions should be cross-checked against the hashed final V3 source listed in the manifest before submission.

The final engine does not read reference poses to estimate motion or select candidates. **Critical audit:** verify from code that quality ranking uses only image geometry, that the frame-gap arbitrary scale is not fitted to reference poses, and that the two candidates' keyframe anchors refer to the same source frame. Save per-frame `status`, `match_source`, `descriptor_valid`, `flow_valid`, `both_valid`, inlier counts and parallax for traceability.

---

## 5. Feature Correspondence Results

### 5.1. Same-implementation sequential frontend comparison [C]

Binary success is defined by **≥8 fitted-model RANSAC inliers**; average inlier density is reported separately.

| Detector | Fixed success | Adaptive-v2 success | Fixed mean inliers | Adaptive mean inliers | Adaptive − fixed | Relative decrease |
|---|---:|---:|---:|---:|---:|---:|
| ORB | 98.766% | 98.766% | 336.18 | 288.22 | −47.96 | 14.3% |
| AKAZE-MLDB | 86.208% | 86.050% | 218.80 | 85.20 | −133.61 | 61.1% |
| SIFT | 99.447% | 99.447% | 294.64 | 173.00 | −121.64 | 41.3% |

Inlier-count paired 95% bootstrap intervals for adaptive minus fixed are **ORB [−86.148, −18.465]**, **AKAZE [−223.596, −56.879]**, and **SIFT [−203.592, −52.657]**. All three exclude zero in this resampling analysis. Binary success is unchanged for ORB and SIFT, while AKAZE decreases by approximately 0.158 percentage points; its success-difference interval [−0.00474, 0] touches zero. These results show that a pass/fail geometric-fit metric can remain high while accepted correspondence density falls substantially. The same-code control reproduced original fixed results for ORB/AKAZE; SIFT showed small numerical differences in 42 ratio-match and 40 inlier-count observations without altered binary success, which should remain documented.

### 5.2. Calibrated non-mutual reference results [C]

**Operating point:** Sampson threshold 2 px; minimum eight reference-consistent matches; scope all 14,365 pairs.

| Detector | Preprocessing | Precision | Recall | F1 | Count-based AP |
|---|---|---:|---:|---:|---:|
| ORB | Raw | .990099 | .262467 | .414938 | .597611 |
| ORB | Fixed CLAHE | .910517 | .863517 | **.886394** | **.930927** |
| ORB | Adaptive v2 | .908248 | .857393 | .882088 | .926415 |
| AKAZE-MLDB | Raw | .991453 | .101487 | .184127 | .337309 |
| AKAZE-MLDB | Fixed CLAHE | .986355 | .442695 | **.611111** | **.828474** |
| AKAZE-MLDB | Adaptive v2 | .991189 | .393701 | .563557 | .814632 |
| SIFT | Raw | .981481 | .278215 | .433538 | .661262 |
| SIFT | Fixed CLAHE | .786182 | .945757 | **.858618** | **.944442** |
| SIFT | Adaptive v2 | .800000 | .913386 | .852941 | .935690 |

The raw policy is highly precise but misses many reference-positive pairs. Fixed CLAHE raises recall dramatically; compared with it, the tested adaptive policy generally trades some correspondence or recall away without an overall ranking benefit at this operating point.

### 5.3. Mutual matching against COLMAP reference [C]

**Operating point:** 2 px, minimum eight mutually matched reference inliers, all pairs.

| Detector | Preprocessing | Precision | Recall | F1 | AP (inlier-count score) | AP (fraction score) |
|---|---|---:|---:|---:|---:|---:|
| ORB | Raw | 1.000000 | .172353 | .294030 | .548765 | .515381 |
| ORB | Fixed CLAHE | .976879 | .739283 | **.841633** | **.918386** | .598424 |
| ORB | Adaptive v2 | .976690 | .733158 | .837581 | .912467 | .587178 |
| AKAZE-MLDB | Raw | .990654 | .092738 | .169600 | .357551 | .331983 |
| AKAZE-MLDB | Fixed CLAHE | .988584 | .378828 | **.547755** | **.815782** | .611731 |
| AKAZE-MLDB | Adaptive v2 | .994737 | .330709 | .496389 | .796251 | .602018 |
| SIFT | Raw | .982578 | .246719 | .394406 | .653586 | **.550602** |
| SIFT | Fixed CLAHE | .836622 | .927384 | **.879668** | **.943451** | .506507 |
| SIFT | Adaptive v2 | .841283 | .895013 | .867317 | .932795 | .504839 |

For SIFT, the fraction-based AP ranking differs from count-based AP, emphasizing that AP is meaningful only together with its scoring function and reference-label definition. We do not use fraction AP as a substitute for count AP.

**Local versus nonlocal mutual F1, fixed → adaptive:**

| Detector | Local, gap ≤100 frames | Nonlocal, gap ≥200 frames |
|---|---|---|
| ORB | .934295 → .929147 | .629630 → .629630 |
| AKAZE-MLDB | .737143 → .677323 | .085603 → .085603 |
| SIFT | .975758 → .960799 | .696398 → .687609 |

The local/nonlocal split matters: the nonlocal task has sparser positive overlap and is closer to long-range loop-candidate retrieval, but no online loop-closing SLAM optimization was performed here.

### 5.4. Paired uncertainty: mutual F1 and AP [C]

The reported differences are **adaptive v2 minus fixed CLAHE**; intervals are unadjusted nominal 95% paired block-bootstrap intervals. All-pair results:

| Detector | ΔF1 | F1 95% interval | ΔCount-AP | Count-AP 95% interval |
|---|---:|---|---:|---|
| ORB | −.00405 | [−.01336, +.00138] | −.005919 | [−.016435, +.000428] |
| AKAZE-MLDB | −.05137 | [−.10383, −.00768] | −.019531 | [−.049669, −.001585] |
| SIFT | −.01235 | [−.03171, −.00117] | −.010656 | [−.023613, −.000844] |

**Local AP intervals:** ORB Δ=−.001394 [−.003772, −.0000879]; AKAZE Δ=−.005702 [−.014513, −.0000515]; SIFT Δ=−.001579 [−.004880, +.000000126]. **Nonlocal AP intervals:** ORB Δ=+.001188 [−.002563, +.007495]; AKAZE Δ=+.007084 [−.002032, +.027248]; SIFT Δ=−.009406 [−.055220, +.004917]. All three nonlocal AP intervals span zero.

**Local F1 intervals:** ORB Δ=−.00515 [−.01624, +.00171]; AKAZE Δ=−.05982 [−.11733, −.00638]; SIFT Δ=−.01496 [−.03200, −.00234]. **Nonlocal F1 intervals:** ORB 0 [0,0]; AKAZE 0 [0,0]; SIFT −.00879 [−.07880,+.00943]. Zero-width intervals should not be misinterpreted as proof of broad algorithmic equivalence; they describe this particular statistic and resampling experiment.

The convergent reduction in all-pair F1 and count AP for **AKAZE-MLDB and SIFT** is the strongest correspondence-level evidence against adaptive v2 here. ORB all-pair intervals include zero. Marginal, nominally nonzero local AP intervals should be read cautiously because multiple detectors, partitions, metrics and thresholds were explored.

### 5.5. Preprocessing overhead [C]

| Method component | Mean latency per image |
|---|---:|
| Raw pass-through | 0.001258 ms |
| Fixed CLAHE | 0.362697 ms |
| Adaptive v2 total | 5.655093 ms |
| Adaptive contrast decision | 5.323949 ms |
| Adaptive enhancement application | approximately 0.329755 ms |

The adaptive variant is approximately **15.6×** slower in preprocessing than fixed CLAHE; the decision statistic accounts for about **94.1%** of adaptive preprocessing time. This is **not** a 15.6× slowdown of the entire visual-odometry pipeline, which was not assessed under the same timing benchmark.

## 6. Visual Odometry Results: Complete Development History

### 6.1. Original engine and first 100-frame pilot [E]

The original video-based monocular VO engine used ORB/AKAZE descriptors, ratio matching, calibrated essential matrices, `recoverPose` and constant arbitrary translation steps. Reusing those functions in the image-sequence runner yielded:

| Configuration | Accepted / 99 | Mean matches | Mean pose-recovery inliers | Mean recovered-inlier fraction |
|---|---:|---:|---:|---:|
| ORB raw | 9 (9.1%) | 1103.22 | 88.52 | .080 |
| ORB CLAHE | 6 (6.1%) | 1113.06 | 60.38 | .054 |
| AKAZE raw | 3 (3.0%) | 731.71 | 22.62 | .029 |
| AKAZE CLAHE | 1 (1.0%) | 1277.45 | 17.27 | .014 |

These inliers are *cheirality-filtered recoverPose inliers*, not the total essential-matrix RANSAC count. The 20% acceptance gate rejected most motions despite plentiful tentative matches. This pilot motivated separate essential, cheirality and parallax diagnostics rather than treating feature matching as VO success.

### 6.2. Adaptive anchoring and separate geometry diagnostics (V2) [E]

A revised image-based engine included anchor gaps, accepted/rejected statuses, essential RANSAC counts, cheirality-valid counts, and parallax diagnostics. On the **first 100 frames**, accepted updates were **46/99 ORB raw**, **43/99 ORB CLAHE**, **47/99 AKAZE raw**, and **47/99 AKAZE CLAHE**. ORB failures were dominated by low cheirality fraction (39 raw; 40 CLAHE), while AKAZE failures were often due to insufficient cheirality inliers (40 raw; 35 CLAHE). These numbers establish changes in the *acceptance protocol*, not necessarily improved accuracy.

The initial COLMAP-relative evaluation on this 100-frame subset had only one accepted associated reference pose for most configurations and three for AKAZE CLAHE. The reported three-point aligned residual **0.0091** reconstruction units for AKAZE CLAHE was explicitly **discarded** as an insufficiently constrained, in-sample alignment result; it must not be cited as algorithm accuracy.

### 6.3. V3 pilot: unit and gap-aware arbitrary translation [E]

The first **600-frame** V3 experiments used unit-scale and gap-aware arbitrary translation choices. Both versions shared the same pose-acceptance decisions:

| Detector/preprocessing | Accepted /599 | Rate |
|---|---:|---:|
| ORB raw | 173 | 28.9% |
| ORB CLAHE | 255 | 42.6% |
| AKAZE raw | 111 | 18.5% |
| AKAZE CLAHE | 147 | 24.5% |

ORB CLAHE showed markedly fewer `few_matches` outcomes (14) than ORB raw (168), while AKAZE raw reported 336 and AKAZE CLAHE 207. The gap-aware scale rule assumes approximate motion magnitude proportional to the frame gap and is **not** measured depth or recovered metric translation. These unit/gap pilot outputs were generated, but a supported **head-to-head held-out RMSE result for unit versus gap** was not supplied in the project conversation; therefore no winner on that metric is claimed.

### 6.4. Optical flow and dual-candidate pose selection [E]

An updated V3 added pyramidal Lucas–Kanade optical-flow tracks with forward–backward checks alongside descriptor matches, and geometric validation for each candidate. Under ORB CLAHE on 600 frames, with minimum essential inliers 15, minimum cheirality inliers 10, minimum cheirality fraction **0.06**, parallax minimum **0.1°**, and anchor gap maximum 20, accepted updates increased from **255/599 (42.6%)** to **351/599 (58.6%)**. The earlier selected-source table showed descriptor chosen on 289 frames with 207 accepted and flow on 310 frames with 144 accepted (plus one initial frame), demonstrating that source-selection frequency alone was not the determinant of reliability.

The dual-source implementation already validated both candidate types; changing the ranking logic did **not** increase acceptance, remaining **351/599**. The diagnostics for this baseline gave **172 low-cheirality-fraction rejections**, **75 insufficient-cheirality-inlier rejections** and **one low-parallax rejection**. These are threshold-trigger categories, not proved physical causes of all failures.

### 6.5. Cheirality-threshold ablation: coverage versus error [E]

Reducing the minimum cheirality fraction from **0.06 to 0.03** (holding other listed parameters fixed) increased accepted updates from **351/599 (58.6%)** to **406/599 (67.8%)**. Fresh COLMAP-indexed accepted poses increased from **14/30 to 16/30**. Under the relaxed threshold, the union of independently accepted candidate sources comprised **340 descriptor-valid**, **160 flow-valid**, **94 valid under both**, hence **406 valid under at least one** by the engine criteria. Counts under 0.03 are *not* externally validated translations.

To compare trajectory behavior, the two runs were evaluated on **12 jointly fresh reference frames**: IDs 60, 120, 140, 160, 180, 200, 280, 360, 440, 480, 520 and 580. The first seven aligned Sim(3), and the final five were held out.

| Metric | Threshold .06 | Threshold .03 |
|---|---:|---:|
| Accepted transitions | 351/599 | 406/599 |
| Fresh reference poses (each run) | 14 | 16 |
| Jointly fresh evaluation frames | 12 | 12 |
| Train / test | 7 / 5 | 7 / 5 |
| Held-out RMSE (reconstruction units) | **1.7171** | 2.3537 |
| Held-out median | **0.8351** | 2.0281 |
| Held-out p95 | **2.9765** | 3.2660 |
| Fitted Sim(3) scale | 0.0176819 | 0.0208262 |

The more permissive threshold achieved **9.2 percentage points** higher tracking acceptance but **approximately 37.1% higher held-out RMSE**. This is an exploratory, five-test-pose comparison and does not provide a robust statistical test of generalization. It does refute the naive inference that acceptance improvements *must* improve trajectory consistency.

### 6.6. Selection-policy ablation: same acceptance, different drift [E]

With the stricter **0.06** threshold and the same 600 frames, source-ranking policies were compared directly. The quality-based approach selected descriptor candidates on **225** accepted transitions and flow on **126**; descriptor-first selected descriptors on **295** and flow on **56**. Both accepted **351/599** transitions and had **14 jointly fresh reference poses**. Sim(3) alignment used earlier frame IDs **60, 80, 120, 140, 160, 180, 200, 280**; testing used **360, 440, 480, 520, 540, 580**.

| Metric | Quality-based | Descriptor-first |
|---|---:|---:|
| Accepted transitions | 351 | 351 |
| Reference poses for alignment / test | 8 / 6 | 8 / 6 |
| Held-out RMSE | **1.6040** | 2.6503 |
| Held-out median | **0.8783** | 2.4815 |
| Held-out p95 | **2.8706** | 3.3543 |
| Fitted Sim(3) scale | 0.0175005 | 0.0191538 |

Quality selection achieved approximately **39.5% lower held-out RMSE** for the same acceptance rate and coverage in this small pilot. This suggests that the **choice of valid relative-pose candidate** can matter for accumulated trajectory consistency, even when neither policy accepts more frames. Because the same sequence informed tuning and only six test poses were used, this is preliminary rather than independent proof of superiority.

### 6.7. Frozen full-sequence ORB–CLAHE V3 [E]

The preferred pilot configuration was frozen and run over **all 3,385 images** with ORB, fixed CLAHE, 1,500 feature budget, dual descriptor/optical-flow candidate evaluation, quality-based selection, essential-inlier minimum 15, cheirality-inlier minimum 10, minimum cheirality fraction 0.06, parallax minimum 0.1°, anchor gap maximum 20, matching ratio 0.75, essential RANSAC threshold 1.5 px, optical-flow point budget 1,800 and forward–backward tolerance 1.5 px. Frame-gap-dependent translation remained arbitrary-scale, and no full SLAM map or loop correction was used.

**Accepted:** **1,188 / 3,384** attempted transitions, **35.1064%**. Rejection counts were **1,227 low cheirality fraction**, **962 insufficient cheirality inliers**, **7 low parallax** (plus the initial frame). The paired-source diagnostics were **1,012 descriptor valid**, **356 flow valid**, and **180 both valid**, yielding **1,188 either valid**; thus **176** accepted transitions were exclusive to optical flow, **832** exclusive to descriptor matching, and **180** shared. This confirms the complementarity of flow without demonstrating metric trajectory quality.

Accepted pose density varied markedly over time. By frame 250, cumulative accepted updates were 150; by frame 500, 295; by frame 1,000, 475; by frame 2,000, 691; and by the end, 1,188. In the 250-transition interval from 1,750 to 2,000, only **34** additional updates were accepted. A first-600-frame acceptance rate of 58.6% therefore overstated the reliability of the same configuration over the full sequence, whose accepted-transition rate was only 35.1%.

**Reference coverage and trajectory consistency:** All **170** COLMAP reference frame IDs were present among the **3,385** VO rows, but only **60/170 (35.294%)** corresponded to *fresh accepted VO updates*. A chronological Sim(3) fit used **36** earlier fresh poses, leaving **24** later fresh poses for held-out error measurement. With the **assumed `twc` reference convention**, the user-reported output was:

| Quantity | Full-sequence value |
|---|---:|
| Images / attempted frame transitions | 3385 / 3384 |
| Accepted transitions | **1188 (35.1064%)** |
| Reference poses / fresh accepted poses | **170 / 60 (35.294%)** |
| Sim(3) calibration poses | 36 |
| Held-out reference poses | 24 |
| Sim(3) scale | 0.0128284071 |
| Held-out translational RMSE | **4.0892926 reconstruction units** |
| Held-out translational median | **3.9457234 reconstruction units** |
| Held-out translational 95th percentile | **4.9862283 reconstruction units** |

The error is computed from **fresh, associated poses**, not a fabricated complete trajectory. Its values should not be expressed in metres; `twc` is not yet independently verified; and the COLMAP-derived reference is not physically measured ground truth. The comparison to the 600-frame pilot RMSE is not a controlled apples-to-apples error change because the temporal horizon, fitted alignment, and held-out samples differ. RPE and rotational trajectory error were **not** established by these outputs.

---

## 7. Cross-Stage Analysis

**Finding 1 — More feature correspondences do not ensure better downstream pose recovery.** The original 100-frame engine frequently produced hundreds to thousands of matches but accepted only 1–9 of 99 relative poses. This demonstrates a separation between tentative descriptor matching and cheirality-validated motion in the tested engine; it does not establish an inherent failure of all underwater VO methods.

**Finding 2 — Image-wise adaptive preprocessing is not free.** The contrast decision skipped CLAHE on nearly one quarter of frames but introduced substantial computation while reducing correspondence counts, and did not improve overall mutual F1/AP relative to fixed CLAHE. A plausible mechanism involves changed keypoint response and descriptor repeatability across differently processed frames. The 16 AKAZE mixed-pair failures support targeted follow-up but do not alone prove the mechanism.

**Finding 3 — Binary pairwise geometric success can hide substantial degradation.** ORB and SIFT fixed-versus-adaptive fitted-model success percentages remain identical despite mean inlier reductions of approximately 14.3% and 41.3%; correspondence redundancy is an independent performance dimension.

**Finding 4 — A permissive acceptance gate can degrade drift.** The 0.03 cheirality fraction admitted 55 more 600-frame transitions than 0.06 but had higher held-out position error on five jointly evaluated reference poses. Thus a claimed 80% tracking target should never be obtained by relabeling weak poses as valid or by minimizing error after fitting on those same poses.

**Finding 5 — Which valid pose is selected matters.** Quality-based versus descriptor-first selection had identical accepted counts but different held-out error across six jointly available test poses. This motivates using geometric support in source selection, while stopping short of asserting generality.

**Finding 6 — Full-sequence behavior differs strongly from short pilots.** The frozen algorithm accepted 58.6% of transitions in the first 600 frames but only 35.1% across all 3,385. Coverage must accompany any trajectory RMSE; a sparse 24-pose test is not an indicator of near-complete localization.

## 8. Threats to Validity

1. **Dataset generalization:** All reported matching and VO experiments use a single archaeological sequence. Adaptive thresholds and VO changes were informed by that same sequence; chronological held-out frames do not provide independent sequence-level validation.
2. **Reference provenance:** COLMAP trajectories and pair labels are offline reconstructions and labels, not independent measurement truth. Pose convention (`Twc`/`Tcw`), quaternion convention, camera center definition and reconstruction scale need explicit verification; this directly affects quoted trajectory errors and even calibrated epipolar geometry if convention assumptions are inconsistent.
3. **Selection bias and coverage:** Trajectory evaluation includes only fresh accepted reference poses. Failed transitions carry forward stale coordinates and long gaps can accumulate drift. The conditional held-out RMSE cannot characterize error on all frames.
4. **VO is not full SLAM:** The current engine has no independently established persistent map, global bundle adjustment, loop closure correction, depth-backed PnP, DVL/inertial scale, or metric translation recovery. Calling the resulting paths full-SLAM trajectories would overclaim.
5. **Two-view degeneracy:** Cheirality inlier counts may fail because of low translation observability, repeated textures, calibration error, imperfect matches, high rotation, or RANSAC sampling. Diagnostic failure codes are not causal proof.
6. **Statistics:** Keyframe pairs are dependent and bootstrapping only first-frame anchors approximates rather than perfectly models dyadic dependence. Most confidence intervals are unadjusted for many comparisons.
7. **Implementation drift:** The frontend correspondence benchmark and the VO engines use different acceptance logic and have evolved across experiments. Direct detector ranking across stages is therefore not a controlled benchmark without harmonized settings.
8. **Latency scope:** The measured preprocessing cost omits detector, matching, optical-flow, essential-matrix and trajectory evaluation overhead. Hardware and OpenCV build details must accompany camera- and platform-specific timing claims.
9. **Exploratory pose experiments:** Five- and six-pose held-out comparisons support hypotheses but cannot establish robust improvements; different 600-frame comparisons can use different common-pose sets and are not directly cross-comparable.
10. **No true cost map:** All generated “cost maps” in the plotting suite are F1/precision/recall response surfaces over Sampson threshold and minimum inlier counts, not spatial traversability, occupancy, depth maps or vehicle navigation costs.

## 9. Reproducibility and Availability

The reproducibility release should record software versions, hardware configuration, seed values, feature/matcher/RANSAC settings, reference-frame conventions, command invocations, raw-file checksums and output CSV schemas. The experimental hierarchy on the researcher's Windows workstation includes:

```text
archaeo_sequence_6_raw_data/
├── raw_data/
│   ├── images_sequence_6/                         # 3,385 grayscale images
│   ├── sequence6_frontend_results/
│   ├── sequence6_calibrated_reference/
│   ├── sequence6_calibrated_feature_results/
│   └── sequence6_mutual_matching_results/
├── sequence6_adaptive_v2_full/
├── sequence6_fixed_clahe_control_full/
├── sequence6_adaptive_v2_calibrated_reference_results/
├── sequence6_adaptive_v2_mutual_matching_results/
├── sequence6_paper_validation/
│   ├── adaptive_reference_f1_bootstrap.csv
│   └── adaptive_v2_mutual_ap_paired_bootstrap.csv
├── colmap_traj_sequence_6.txt
├── new_archaeo_colmap_traj_sequence_06.txt
├── colmap_detected_loop_sequence_6.txt
├── aqualoc_vo_v3_pilot/
└── paper_research_draft/
    ├── scripts/                                    # Dedicated plotting modules
    ├── figures/
    ├── vo_pilot_unit/
    ├── vo_pilot_gap/
    ├── vo_pilot_optical_flow/
    ├── vo_pilot_dual_source/
    ├── vo_pilot_cheirality_ablation/
    ├── vo_pilot_descriptor_first/
    ├── cheirality_threshold_comparison/
    ├── vo_selection_comparison/
    ├── vo_full_sequence_v3/
    └── full_sequence_trajectory_evaluation/
```

Some earlier artifacts may reside under `Downloads/aqualoc_6_v1` or alongside the root instead of the shown canonical path; the actual directory inventory should be checked before archiving. The figure filenames in §13 are **expected output names from the scripts**, not independent evidence that every PNG already exists on the user's computer.

### Suggested release metadata

- Camera calibration and original timestamps; path of the exact COLMAP pose export chosen as reference; reference overlap label provenance.
- Six original + three adaptive calibrated and mutual tables; raw/adaptive/fixed frontend control outputs; pairing manifest; associated match-coordinate archives.
- V1 and V2 adaptive decisions and thresholds; bootstraps and metadata; original fixed–adaptive timing logs and repetitions.
- VO original engine, image-sequence runner, final V3 code, frozen command, all pilot and full-sequence trajectories, status flags and evaluator versions.
- README documenting reference convention verification, selected-pose coverage policy, numerical precision, and exploratory nature of within-sequence tuning.

## 10. Conclusion

A comprehensive, single-sequence evaluation of AQUALOC Sequence 06 establishes that fixed CLAHE is a strong baseline for classical feature matching, whereas the tested frame-wise contrast-adaptive switching strategy offers no overall correspondence benefit and introduces greater preprocessing cost. The effect is visible across frontend inlier density and COLMAP-relative precision/recall/F1/AP comparisons, with paired bootstrap intervals indicating overall declines for AKAZE-MLDB and SIFT. Exploratory visual odometry extends the analysis beyond feature correspondences: better matching coverage and more accepted essential-matrix decompositions do not inherently produce lower accumulated trajectory error. Optical flow complemented descriptor matching, yet the frozen full-sequence monocular system accepted only 35.1% of transitions and yielded a held-out Sim(3) positional RMSE of 4.089 reconstruction units on 24 fresh reference poses under an unverified `Twc` assumption. The study thus argues for multi-stage evaluation using calibrated correspondence metrics, pose observability, tracking coverage, held-out drift and computation. Its negative adaptive-CLAHE finding is sequence-specific, while its VO comparisons remain exploratory. Independent reference-convention confirmation and cross-sequence evaluation are necessary before stronger localization or generalization claims.

## 11. Future Work

The next experiments should prioritize: (i) independently confirming COLMAP `Twc/Tcw` and scale; (ii) repeating the *frozen* comparison on at least one other AQUALOC sequence; (iii) locally evaluating the same test frames for metric/scale-consistent rotational and translational RPE, including clear failure coverage; (iv) measuring continuous tracking segment lengths, maximum staleness, and time-varying drift; (v) comparing sequence-level temporal decision hysteresis under a prespecified threshold, explicitly as a new hypothesis; and (vi) incorporating genuinely synchronized IMU/pressure or registered depth for metric motion constraints, if validated sensor synchronization and calibration support it. No such results are included in the current article.

## 12. Submission Readiness: Evidence-Gated Verification

**No figures are inserted in this draft.** The table below distinguishes checks prepared in the manuscript from those **not yet executed** against the researcher's frozen local sources. A publication checklist cannot legitimately be checked off merely by adding text or supplying unrun code.

| Submission requirement (excluding insertion of figures) | Status at manuscript v2.2 | Evidence / next action |
|---|---|---|
| Correct scientific terminology, VO vs SLAM, reference labels and units | **COMPLETE IN TEXT** | Every trajectory error is reconstruction/export-relative; no claim of independently measured metre-scale ATE; heatmaps are parameter-response surfaces |
| Contemporary relevant literature and comparison table | **COMPLETE IN TEXT** | §2.1, references [9]–[16] with DOI/publisher records; no cross-system ranking invented |
| Target venue and author/ethics/data template sections | **DRAFTED** | Provisional *Sensors* (MDPI); complete actual affiliations, contributor confirmation, funding/conflict declarations, repository deposit and official template submission |
| AP score definitions, threshold formulas and classification semantics | **CODE AND TEXT PREPARED** | §4; run the frozen-data audit and resolve every mismatch before checking off |
| SHA-256 input/output manifest and software/hardware environment | **SCRIPT PREPARED — NOT RUN ON USER'S PC** | `scripts/audit_frozen_results.py`; inspect `submission_audit/audit_report.md`, `reproducibility_manifest.json`, `environment.json` |
| Exact local COLMAP pose semantics, quaternion rotation convention, metric scale and IMU offset | **PARTIALLY VERIFIED FROM ORIGINAL DATASET PAPER; LOCAL EXPORT UNRESOLVED** | §3.3; original dataset paper identifies field order and pressure scaling, but both local exports require a primary export/code provenance check. Use `reference_pose_rpe_audit.py` for a **sensitivity check only** |
| Source-matched table/figure numerical re-audit | **NOT EXECUTED HERE** | Required original CSVs are not available in the authoring container; run checksum audit and inspect every failed check |
| Alternative temporal bootstrap block sizes | **REPRODUCTION COMMAND PREPARED — NOT RUN** | Use saved `paired_ap_bootstrap.py --blocks 1 5 10 20` on local pair data; export/compare CI-by-block size; first-frame blocks do not remove dyadic dependence |
| Image-level/dyadic resampling sensitivity | **NOT COMPLETED** | Endpoint-aware resampling method and implementation need independent validation; must not claim interval family-wise significance |
| Frozen-policy out-of-sequence validation | **NOT COMPLETED** | No second raw sequence or verified cross-sequence run in this authoring environment; execute the prespecified protocol in `INDEPENDENT_SEQUENCE_PROTOCOL.md` |
| RPE orientation and translation evaluation | **SCRIPT PREPARED — NOT RUN ON USER DATA** | `reference_pose_rpe_audit.py` writes fresh held-out fixed-gap RPE per convention with pair count, but is only meaningful after pose semantics are confirmed |
| Author approvals, affiliation, publication data/code archive, funding and conflict disclosures | **AUTHOR ACTION REQUIRED** | Submitter must provide truthful declarations and stable archive links; text cannot certify personal claims |
| Publisher template, citation order, journal-specific figure files | **PENDING FORMATTING/IMAGES** | Convert Markdown to current *Sensors* LaTeX/Word template after validation; inserting verified images is expressly deferred |

### 12.1. Local read-only audit commands (PowerShell)

Extract this submission-preparation package into `archaeo_sequence_6_raw_data` (its folder may be renamed, but preserve `scripts/`). The audit does not modify any existing experiment output and returns a nonzero exit code when assertions fail.

```powershell
$root = 'C:\Users\Raaghav\Downloads\archaeo_sequence_6_raw_data'
$py = "$env:USERPROFILE\aqualoc-venv\Scripts\python.exe"
$pkg = "$root\AQUALOC_Submission_Preparation_V2_2"
& $py "$pkg\scripts\audit_frozen_results.py" --root $root --hash-images
& $py "$pkg\scripts\reference_pose_rpe_audit.py" --root $root --rpe-gap 20
notepad "$root\paper_research_draft\submission_audit\audit_report.md"
```

For temporal-sensitivity experiments, first preserve the one-block-size AP output if present, then run the **original** saved score-validated AP bootstrap with several block lengths. Results are exploratory and the dyadic shared-endpoint dependence remains:

```powershell
$base = "$root\sequence6_paper_validation"
Copy-Item "$base\adaptive_v2_mutual_ap_paired_bootstrap.csv" "$base\adaptive_v2_mutual_ap_paired_bootstrap_block10_original.csv" -ErrorAction SilentlyContinue
& $py "$root\aqualoc_final_ap_validation\paired_ap_bootstrap.py" --root $root --reps 1000 --blocks 1 5 10 20 --seed 20261104
```

**Operating-point and source audit:** At 2 px, positive prediction means ≥8 reference-consistent (or mutual reference-consistent) inliers, not ≥8 raw descriptor matches. Binary F1 is calculated using all 14,365 keyframe pairs per detector/method; AP is continuous ranking using count or normalized fraction as explicitly labeled. Recomputed values must match the manuscript tables to 1e−5 or the associated tables must be corrected from sources with an audit log. The source-to-output manifest must be archived and checksum-locked **after** any correction.

**Reference conventions:** The official AQUALOC publication identifies the custom 8-field format and says pressure was used for scaling the public reconstruction baseline. It does **not** establish that either of the two *different-scale* local text copies retained the original pressure-calibrated metric scale or whether their `tx,ty,tz` columns are directly camera centers. COLMAP `images.txt` separately uses world-to-camera quaternion `qw,qx,qy,qz`; do not assume the AQUALOC export uses that rotation direction. Source exporter code or official dataset README and a pose sanity check are required. The reference-sensitivity script reports both possibilities but **must not** select the convention with smaller positional error.

**Statistical reporting:** All published bootstrap confidence intervals are **unadjusted nominal 95% within-sequence intervals**, based on a one-endpoint temporal block scheme. A bootstrap interval excluding zero is not equivalent to adjusted statistical significance over all metrics, detectors and thresholds. Disclose the v2 contrast threshold and VO quality choices made after examining Sequence 06. Do not claim external validity without out-of-sequence tests.

**No unseen results:** No corrected-geometry ATE, RPE, multiple-block sensitivity table, independent AQUALOC evaluation or data repository DOI has been observed in this conversation. The scripts are reproducibility deliverables, **not** evidence that these tests already passed.

---

## 13. Ten Recommended Paper Figures From `paper_research_draft`

The ten items below are **specific expected PNG names produced by previously supplied scripts**. Paths are relative to `archaeo_sequence_6_raw_data/paper_research_draft/`. These files were not uploaded here; their existence and graphical integrity on the local Windows machine are **not verified in this drafting session**. If a file is absent, regenerate using the named script and confirm it represents the cited experiment. Recommended placement follows the manuscript's argument from correspondence through trajectory.

| Fig. | Expected PNG path (inside `paper_research_draft/`) | Manuscript placement | Scientific caption / intent | Generator |
|---:|---|---|---|---|
| **1** | `figures/02_feature_maps_matches/keypoints_ORB_clahe_frame100.png` | §4.1 | **ORB features after fixed CLAHE.** Representative detector response on image frame 100; qualitative visualization, not a per-sequence accuracy metric. | `scripts/02_feature_maps_matches.py` |
| **2** | `figures/02_feature_maps_matches/inlier_matches_ORB_clahe_100_120.png` | §4.1 | **Illustrative ORB correspondence geometry, frames 100 and 120.** Freshly recomputed fitted-F inliers for visualization; *not* COLMAP reference inliers, so do not label as calibrated reference positives. | `scripts/02_feature_maps_matches.py` |
| **3** | `figures/03_parameter_response_heatmaps/calibrated_all_pairs_f1_response_surface.png` | §5.2 | **Calibrated F1 response surfaces.** Detector × preprocessing grid over Sampson threshold and minimum reference-inlier count (all pairs); these are evaluation-parameter heatmaps, not navigation costs. | `scripts/03_cost_maps_heatmaps.py` |
| **4** | `figures/05_publication_charts/mutual_f1_all_pairs_bar_comparison.png` | §5.3 | **Mutual-reference F1 for raw, fixed CLAHE, and adaptive v2** across ORB, AKAZE-MLDB and SIFT at 2 px and eight inliers. | `scripts/05_publication_bar_charts.py` |
| **5** | `figures/04_adaptive_vs_fixed/mutual_average_precision_bootstrap_ci.png` | §5.4 | **Paired 95% bootstrap intervals for adaptive minus fixed count-based AP**, across detectors and pair scopes; zero line indicates no difference. | `scripts/04_adaptive_vs_fixed.py` |
| **6** | `figures/05_publication_charts/mutual_average_precision_count_all_pairs_bar_comparison.png` | §5.3–5.4 | **Mutual count-based average-precision comparisons** for detector × preprocessing policy, avoiding score-definition ambiguity. | `scripts/05_publication_bar_charts.py` |
| **7** | `cheirality_threshold_comparison/heldout_error_comparison.png` | §6.5 | **Pilot held-out trajectory error versus reference frame ID**, cheirality 0.06 versus 0.03, seven calibration and five test poses. | `compare_v3_cheirality.py` |
| **8** | `vo_selection_comparison/heldout_error.png` | §6.6 | **Quality-based versus descriptor-first pose selection**, equal acceptance, six held-out COLMAP reference poses; show sample size on figure. | `aqualoc_v3_accuracy_experiment/compare_v3_selection.py` |
| **9** | `full_sequence_trajectory_evaluation/02_tracking_timeline.png` | §6.7 | **Full 3,385-frame acceptance/rejection timeline.** State explicit full-sequence acceptance of 1188/3384, and identify repeated/held poses separately. | `aqualoc_full_sequence_eval/evaluate_full_v3.py` |
| **10** | `full_sequence_trajectory_evaluation/03_heldout_position_errors.png` | §6.7 | **Full-sequence held-out Sim(3)-aligned positional residual against COLMAP**, evaluated on 24 fresh associated poses after fitting on 36; label units as COLMAP reconstruction units, **not metres**. | `aqualoc_full_sequence_eval/evaluate_full_v3.py` |

**Other useful (supplemental) outputs:** `full_sequence_trajectory_evaluation/04_trajectory_projections.png`, `01_reference_coverage.png`, `07_rejection_reasons.png`, `08_parallax_diagnostics.png`, `figures/03_parameter_response_heatmaps/mutual_nonlocal_gap_ge_200_f1_response_surface.png`, and adaptive-versus-fixed preprocessing contrasts. A physical 2D/3D environmental cost map or point-cloud reconstruction has **not** been measured; do not invent one merely for illustration.

### Supplemental figure recommendations (not part of the ten main figures)

The following paths are **candidate or expected local outputs** relative to `archaeo_sequence_6_raw_data/paper_research_draft/`, not files independently verified in this authoring environment. Use them only after inspecting the actual images, checking their generator and confirming the corresponding data and parameters.

| Supplemental item | Expected PNG path under `paper_research_draft/` | Interpretation and caption safeguard |
|---|---|---|
| S1 — Path projection | `full_sequence_trajectory_evaluation/04_trajectory_projections.png` | Sim(3)-aligned VO versus COLMAP XY/XZ/YZ projections. Include reference pose convention, fit interval and the number of fresh tested poses; plots are reconstruction-relative, not metric motion capture. |
| S2 — Pose coverage | `full_sequence_trajectory_evaluation/01_reference_coverage.png` | Show where the 60/170 fresh accepted reference frames occur; distinguish accepted, stale/carried and unavailable poses. |
| S3 — Failure mechanism | `full_sequence_trajectory_evaluation/07_rejection_reasons.png` | Frequency of diagnostic status flags in full 3,384 frame transitions; cheirality-fraction versus inlier-count reasons describe the stopping gate, not necessarily a unique physical cause. |
| S4 — Motion observability | `full_sequence_trajectory_evaluation/08_parallax_diagnostics.png` | Observed parallax and acceptance/rejection patterns. Identify whether estimates are at frame or anchor-pair level, and clarify that low parallax can manifest as a cheirality failure. |
| S5 — Nonlocal overlap | `figures/03_parameter_response_heatmaps/mutual_nonlocal_gap_ge_200_f1_response_surface.png` | Nonlocal mutual-matching F1 response to Sampson threshold and minimum-inlier count; **not a physical robot cost map**. |
| S6 — Preprocessing illustrations | Corresponding raw, fixed CLAHE and adaptive-v2 frames or feature-map comparisons produced under `figures/02_feature_maps_matches/` | Use identical source frame IDs and annotated preprocessing decisions; ensure adaptive-v2 switching is computed from the original frozen decision file and is not cosmetically simulated. |

**Absent results must remain absent:** No environmental 2D/3D traversability cost field, registered 3D map, point-cloud reconstruction, metric absolute trajectory truth or independent loop-closure-corrected SLAM map has been established in the completed experiments. Do not create illustrative results and present them as measured outputs.

**Minimal local inventory (PowerShell):**

```powershell
$root = 'C:\Users\Raaghav\Downloads\archaeo_sequence_6_raw_data'
$draft = Join-Path $root 'paper_research_draft'
Get-ChildItem $draft -Recurse -File -Filter '*.png' |
  Select-Object FullName, Length |
  Format-Table -AutoSize
```

**Regenerate the four correspondence figure families if missing:**

```powershell
$py = "$env:USERPROFILE\aqualoc-venv\Scripts\python.exe"
& $py "$draft\scripts\02_feature_maps_matches.py" --root $root --frame-a 100 --frame-b 120
& $py "$draft\scripts\03_cost_maps_heatmaps.py" --root $root
& $py "$draft\scripts\04_adaptive_vs_fixed.py" --root $root
& $py "$draft\scripts\05_publication_bar_charts.py" --root $root
```

The trajectory figure files in rows 7–10 are produced by the corresponding **previously executed comparison/evaluation tools**; rerun those evaluators only if absent, retaining their original experiment inputs and documented COLMAP convention assumption. Do not regenerate threshold ablations through a different engine configuration and silently reuse the same caption.

---

## Submission Declarations (draft for provisional *Sensors* target)

**Author information:** B. Raaghavan Mookiah. **Affiliation, corresponding-author email, ORCID and coauthor list:** *[AUTHOR TO COMPLETE AND APPROVE — do not invent]*.

**Author Contributions (CRediT):** *[AUTHOR(S) TO VERIFY]* — methodology, software, formal analysis, investigation, visualization, writing—original draft, writing—review and editing, data curation, supervision, funding acquisition where applicable. Do not ascribe work or approvals without explicit author confirmation.

**Funding:** *[AUTHOR TO DECLARE ACTUAL FUNDING, OR CONFIRM “This research received no external funding.”]*

**Institutional Review Board Statement:** Not applicable **if** the study uses only previously published underwater sensor data and no human/animal subjects under the journal's definitions; **author must confirm**.

**Informed Consent Statement:** Not applicable **if** no identifiable human subjects or private patient data are used; **author must confirm**.

**Data Availability Statement:** The original AQUALOC dataset is publicly documented by Ferrera et al. [1] at http://www.lirmm.fr/aqualoc/ (verify the live download location and redistribution terms before deposit). Analysis results, source scripts, trajectory exports, audit manifests and configuration logs are **not yet identified by a stable public repository URL/DOI**. Replace this paragraph with the final verified archive URL and released file inventory before submission. Do not claim files are public when they have not been uploaded.

**Code Availability:** Source benchmark, adaptive preprocessing, VO and evaluation scripts are presently held in the researcher's local project. *[AUTHOR TO INSERT stable code archive URL and immutable release hash once deposited]*.

**Conflicts of Interest:** *[AUTHOR TO CONFIRM actual declaration; do not silently assert none]*.

**Acknowledgments and AI-assisted manuscript preparation:** The manuscript was drafted and edited with assistance from OpenAI ChatGPT for technical writing, code development and experiment interpretation; the author(s) must verify the exact tool/version disclosures required by the selected journal, critically review all generated material and take full responsibility for the work. No claim is made that generative AI independently performed or validated the experiments.

**Software/tool reporting:** AQUALOC dataset [1], OpenCV, NumPy, pandas, Matplotlib and the algorithms cited herein. Exact versions, Windows CPU/GPU/RAM and OpenCV build/seed metadata are exported by the local audit script and must be pasted into a final environment table before submission.

**Venue preparation note:** This is a Markdown working manuscript with required scientific sections and declarations, **not** an official formatted *Sensors* submission file. Adapt to the journal's current article template and verify final author instructions immediately before uploading. Publisher policy requests clear data availability, author contributions, conflict disclosures and disclosure of AI assistance when used.

---

## References

[1] M. Ferrera, V. Creuze, J. Moras, and P. Trouvé-Peloux, “AQUALOC: An underwater dataset for visual–inertial–pressure localization,” *International Journal of Robotics Research*, vol. 38, no. 14, pp. 1549–1559, 2019. DOI: [10.1177/0278364919883346](https://doi.org/10.1177/0278364919883346).

[2] D. G. Lowe, “Distinctive image features from scale-invariant keypoints,” *International Journal of Computer Vision*, vol. 60, no. 2, pp. 91–110, 2004. DOI: [10.1023/B:VISI.0000029664.99615.94](https://doi.org/10.1023/B:VISI.0000029664.99615.94).

[3] E. Rublee, V. Rabaud, K. Konolige, and G. Bradski, “ORB: An efficient alternative to SIFT or SURF,” *IEEE International Conference on Computer Vision (ICCV)*, pp. 2564–2571, 2011. DOI: [10.1109/ICCV.2011.6126544](https://doi.org/10.1109/ICCV.2011.6126544).

[4] P. F. Alcantarilla, J. Nuevo, and A. Bartoli, “Fast explicit diffusion for accelerated features in nonlinear scale spaces,” *British Machine Vision Conference (BMVC)*, 2013. DOI: [10.5244/C.27.13](https://doi.org/10.5244/C.27.13).

[5] K. Zuiderveld, “Contrast limited adaptive histogram equalization,” in *Graphics Gems IV*, P. S. Heckbert, Ed. Morgan Kaufmann, 1994, pp. 474–485. [Bibliographic record](https://gameenginegems.com/gemsdb/article.php?id=970).

[6] J. L. Schönberger and J.-M. Frahm, “Structure-from-Motion Revisited,” *IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, pp. 4104–4113, 2016. DOI: [10.1109/CVPR.2016.445](https://doi.org/10.1109/CVPR.2016.445).

[7] B. D. Lucas and T. Kanade, “An iterative image registration technique with an application to stereo vision,” *International Joint Conference on Artificial Intelligence (IJCAI)*, 1981. [IJCAI proceedings](https://ijcai.org/proceedings/1981/127).

[8] J. Sturm, N. Engelhard, F. Endres, W. Burgard, and D. Cremers, “A benchmark for the evaluation of RGB-D SLAM systems,” *IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)*, pp. 573–580, 2012. DOI: [10.1109/IROS.2012.6385773](https://doi.org/10.1109/IROS.2012.6385773).

[9] M. Leonardi, A. Stahl, E. F. Brekke, and M. Ludvigsen, “UVS: underwater visual SLAM—a robust monocular visual SLAM system for lifelong underwater operations,” *Autonomous Robots*, 47, 1367–1385, 2023. https://doi.org/10.1007/s10514-023-10138-0.

[10] Z. Xin, Z. Wang, Z. Yu, and B. Zheng, “ULL-SLAM: underwater low-light enhancement for the front-end of visual SLAM,” *Frontiers in Marine Science*, 10, 1133881, 2023. https://doi.org/10.3389/fmars.2023.1133881.

[11] J. Drupt, A. I. Comport, C. Dune, and V. Hugel, “MAM³SLAM: Towards underwater-robust multi-agent visual SLAM,” *Ocean Engineering*, 302, 117643, 2024. https://doi.org/10.1016/j.oceaneng.2024.117643.

[12] D. Xiang et al., “An Improved Underwater Visual SLAM through Image Enhancement and Sonar Fusion,” *Remote Sensing*, 16(14), 2512, 2024. https://doi.org/10.3390/rs16142512. **Complete author list to recheck from the publisher record before final styling.**

[13] G. Chen et al., “An underwater visual SLAM system with adaptive image enhancement,” *Ocean Engineering*, 326, 120896, 2025. https://doi.org/10.1016/j.oceaneng.2025.120896.

[14] J. M. Summers, M. W. Jones, and C. Seale, “Impact of Underwater Image Enhancement on Feature Matching,” *Sensors*, 25(22), 6966, 2025. https://doi.org/10.3390/s25226966.

[15] C. Li, S. Anwar, J. Hou, R. Cong, C. Guo, and W. Ren, “Underwater Image Enhancement via Medium Transmission-Guided Multi-Color Space Embedding,” *IEEE Transactions on Image Processing*, 30, 4985–5000, 2021. https://doi.org/10.1109/TIP.2021.3076367.

[16] K. An et al., “DFSVO: Detection-Free Stereo Visual Odometry for underwater localization and mapping,” *Ocean Engineering*, 362, 126244, 2026. https://doi.org/10.1016/j.oceaneng.2026.126244.

**Additional technical sources (not counted as performance comparators):** AQUALOC primary paper (reference [1]); [COLMAP native output conventions](https://colmap.github.io/format.html); [MDPI publication ethics](https://www.mdpi.com/ethics) and [MDPI author layout guidance](https://www.mdpi.com/authors/layout). These are provenance and compliance sources, not novel experimental evidence.

---

## Appendix A. Exact Research-Iteration Record

| Stage | Trial or implementation | Output/result | Reason for the test | Evidence boundary |
|---|---|---|---|---|
| A0 | Timestamp/sensor/calibration audit | 3385 images, ~20 Hz, pinhole+radtan; downloaded IMU camera/noise files | Establish synchronization, sampling, image geometry | IMU was not fused |
| A1 | First 100-image frontend | 1644 pair-config assessments | Feasibility check before full compute | Not independent geometry |
| A2 | Full uncalibrated frontend | 60,774 rows, ORB/AKAZE/SIFT × raw/CLAHE × stride 1/5/20 | Compare temporal match retention | Fitted F from candidate matches |
| A3 | COLMAP files and pair manifest | 170 poses, 14,365 unordered pairs, 170² loop labels | Provide reference geometry and overlap labels | Reconstruction proxy |
| A4 | Calibrated non-mutual | 86,190 original + 43,095 adaptive records | Independent-within-dataset geometry scoring | Pose-convention audit outstanding |
| A5 | Mutual matching | 86,190 original + 43,095 adaptive records | Test bidirectional descriptor consistency | Same images and labels |
| A6 | Adaptive v1, v2 and same-code fixed control | 808 raw / 2577 CLAHE; 30,387 pairs per adaptive or fixed control | Test adaptive switching without implementation confounding | v2 tuned on same sequence |
| A7 | Frontend block bootstrap and runtime | 1,000 reps; inlier differences; 5.655 vs 0.363 ms | Quantify uncertainty and overhead | Preprocessing only; approximate dependence |
| A8 | Mutual F1/AP block bootstraps | 1,000 reps; detector × scope effects | Quantify ranking/threshold stability | Multiplicity and dependence limits |
| B1 | Original video VO engine, Sequence 06 image runner | 1–9 /99 accepted, 4 configurations | Test whether dense matching yields poses | Weak tracking; arbitrary scale |
| B2 | Calibrated VO V2 with anchors | 43–47 /99 accepted | Diagnose pose acceptance and parallax | Changed acceptance protocol |
| B3 | V3 unit/gap scaling pilots | ORB CLAHE 255/599, AKAZE CLAHE 147/599 | Test anchor-length translation assumptions | No reported unit-v-gap error winner |
| B4 | V3 descriptor+optical flow | ORB CLAHE 351/599 | Recover correspondences when descriptors are insufficient | Not independently proven pose correctness |
| B5 | Dual-source selection | Quality acceptance 351/599 | Compare independently validated candidate poses | Same acceptance, policy affects path |
| B6 | Cheirality 0.06 vs 0.03 | 351→406/599; RMSE 1.717→2.354 on five tests | Detect acceptance/accuracy tradeoff | Exploratory; tiny holdout |
| B7 | Quality vs descriptor-first | Same 351/599; RMSE 1.604 vs 2.650 on six tests | Isolate selection effect at constant acceptance | Exploratory; tiny holdout |
| B8 | Frozen full Sequence 06 V3 | 1188/3384 accepted; 60/170 fresh reference poses; 24-test RMSE 4.089 | Measure long-horizon trajectory reliability | Assumed Twc, COLMAP units |

**End of manuscript v2.2 — subject to evidence-gated author approval and local audit.**

## Appendix B. Scientific Evidence Ledger and No-Leakage Requirements

| Evidence | Availability | Can be asserted? |
|---|---|---|
| Sequence 06 user-run counts, metrics, CI tables and VO logs | Transcribed from reported experimental outputs | Yes, labeled researcher-reported and pending frozen-CSV re-audit |
| AQUALOC paper dataset format, pressure-scaled reference and frame stride | Verified from original published dataset paper | Yes, with distinction from the two local exports |
| Local sequence's exact `Twc/Tcw`, quaternion meaning and pressure-scale preservation | Source/README exporter proof absent | **No**; must remain conditional |
| 4.089 held-out Sim(3) positional RMSE | Researcher-reported, assumed `twc` | Yes as exploratory export-relative score, not physical ground truth |
| AP/F1 source file checksum report | Script delivered; user-data execution pending | **No** until local audit passes |
| Alternative bootstrap block-length estimates | Script invocation supplied; not executed | **No** |
| Dyadic endpoint resampling | Not implemented/validated | **No** |
| Independent AQUALOC sequences and stable repository deposit | Not supplied | **No** |

All model/policy choices must be frozen **before** an independent sequence is inspected. Every independent sequence attempted, failure and incomplete run must be recorded; absence of a successful trajectory is itself a result. Continued Sequence 06 parameter search cannot be used as held-out external validation.
