# Discussion

This section interprets the results from experiments, examines specific failure conditions, and discusses the limitations and design rationale behind the pipeline.

## Result interpretation

**Heuristic vs. MiDaS.**

The heuristic method outperformed MiDaS across every metric and every individual video by a large, consistent gap (2.1x lower Euclidean error, 2.0x lower relative spatial error, 4.4x lower trajectory step distance). Here are some reasons explaining why.

**MiDaS's trajectory step distance (14.22 units/frame) is well above what a real player can physically achieve** in a single frame. Assuming standard CS2 running speed (~250 units/second) and a ~30fps extraction rate, plausible per-frame movement is capped around 8–9 units/frame. Heuristic's step distance of 3.21 is reasonably within this range, while MiDaS's 14.22 is not. This indicates MiDaS-based trajectories are not primarily tracking real player movement. This elevated number likely reflects frame-to-frame depth-estimation noise from MiDaS producing slightly different depth values for the same physical scene between adjacent frames.

Parameter tuning aimed at MiDaS's normalization (`depth_window`, `low_pct`, `high_pct`, `fov_scale`) did produce a real, measurable improvement (mean calibration error 122.97 → 116.20). However, even at this tuned configuration, MiDaS's error remains substantially higher (~58% higher) than the tuned heuristic method's (73.54). The gap in full evaluation (99.44 vs. 46.92 mean Euclidean error) is larger than what tuning alone could close. This suggests that while normalization tuning makes an improvement in performance, it does not address the dominant source of MiDaS's error, which is likely frame-to-frame estimation noise than a fixable calibration problem.

Per-video scale factors, fit independently for each method during evaluation, showed comparable variability between the two approaches: heuristic ranged from 4.275–6.174 (a 44% spread relative to its minimum) and MiDaS ranged from 1.145–1.648 (a 44% spread). This contrasts with an earlier analysis pass that had suggested worse scale instability for MiDaS specifically. Although after refinements to the evaluation pipeline (relabeled tracking identities, corrected depth normalization), scale factors appear not to be a meaningful distinguishing feature between the two methods.

All together, these results support a specific claim:

!!! tip ""
    For this task, a simple geometric heuristic approach based on bounding-box size is a more reliable depth extraction method than a general-purpose monocular depth model, primarily due to the learned model's frame-to-frame instability rather than a difference in achievable calibration accuracy.

**Track vs. no track.**

Tracking greatly improved both positional accuracy metrics: mean Euclidean error dropped from 71.91 (no track) to 46.92 (tracked), and mean relative spatial error dropped from 53.06 to 33.18. Trajectory consistency is undefined without tracking, since it requires persistent identity across frames.

The size of tracking's benefit varied significantly by round. In five of six videos, removing tracking increased mean Euclidean error by 10–45%. In `match_2_round_2`, however, error increased by over 300% (25.69 → 102.99). This is the largest gap observed, and notable because this round was the best-performing round for every tracked configuration evaluated in this project. A plausible explanation is that `match_2_round_2` involves more closely-spaced players than other rounds (therefore flagged as having the lowest error under tracking, where persistent identity correctly resolves closely-spaced players that per-frame Hungarian matching, used in the untracked experiment, is more likely to mismatch). This is consistent with, though not directly confirmed by, the occlusion analysis below. This connection has not been tested but could be done in future works (e.g. comparing occlusion rates specifically for `match_2_round_2` against the dataset average).

!!! warning "Evaluation asymmetry"
    As noted in [Results](results.md#track-vs-no-track), the untracked baseline lacks the manual exclusion filtering (POV-self, dead bodies) applied to the tracked evaluation. The reported gap, and particularly the `match_2_round_2` outlier, should be read as an upper bound on tracking's benefit.

## Failure Cases
Three challenging conditions were flagged programmatically as a part of the research for this project:

- **Occlusion**: frames where two or more detected bounding boxes overlap above an IoU threshold (0.3), indicating players close together or partially blocking one another.
- **Motion blur**: frames with low Laplacian variance (below 100.0) in the extracted image, indicating fast camera or subject motion blurring detail.
- **Rapid camera movement**: frames where the synchronized camera yaw changed by more than 5° from the previous frame (wraparound-corrected at the ±180° boundary), indicating a fast pan/turn.

!!! note "Rapid camera movement"
    Camera yaw threshold (5°) might be different if original video was extracted at another frame rate.

### Occlusion

| Occluded | Heuristic mean | Heuristic median | MiDaS mean | MiDaS median | Frame count |
|---|---:|---:|---:|---:|---:|
| False | 47.00 | 29.50 | 100.40 | 83.42 | 11,539 |
| True | 45.32 | 26.19 | 79.95 | 56.70 | 570 |

Unexpectedly, occluded frames showed **lower** mean and median error than non-occluded frames for both methods, contrary to the initial hypothesis. The effect was proportionally larger for MiDaS (a 20% drop) than heuristic (a 4% drop), but the direction was consistent across both.

A plausible explanation specific to this game is that CS2 renders a distinct outline around a player's silhouette when they are occluded behind in-game architecture or special effects (e.g. smoke, flash), so that the player remains visually identifiable to the viewer even when not directly visible. This outline likely allows adequate performance from the detector even under occlusion, unlike occlusion in typical unconstrained video, where a partially hidden object usually degrades detection quality. 

If so, the occlusion flag in this dataset may be capturing a condition the game itself already compensates for visually. However, this failure case also accounts for occlusion behind other players and game UI elements, as well as being partially out of frame, which are not compensated by the silhouette outline. This result should be treated as suggestive rather than conclusive.

The qualitative example frames below are used to inspect specific occluded cases in addition to the aggregate statistics.

<div class="image-carousel">

  <img src="../../assets/images/match_1_round_1_frame001839_occluded_worst.png" alt="Occlusion worst, match 1 round 1">
  <img src="../../assets/images/match_1_round_1_frame000630_occluded_low_error_despite_condition.png" alt="Occlusion low error, match 1 round 1">
  <img src="../../assets/images/match_1_round_2_frame000207_occluded_worst.png" alt="Occlusion worst, match 1 round 2">
  <img src="../../assets/images/match_1_round_1_frame002280_occluded_low_error_despite_condition.png" alt="Occlusion low error, match 1 round 1">
  <img src="../../assets/images/match_3_round_2_frame001800_occluded_worst.png" alt="Occlusion worst, match 3 round 2">
  
</div>

### Motion Blur

| Blurry | Heuristic mean | MiDaS mean | Frame count |
|---|---:|---:|---:|
| False | 46.92 | 99.44 | 12,109 |

No frames in the evaluated dataset were flagged as blurry under the configured threshold (Laplacian variance < 100.0) for either method. This could reflect either a genuine absence of significant motion blur in this broadcast-style footage, or a threshold poorly calibrated to this dataset's actual blur characteristics. Given the absence of flagged frames, motion blur could not be evaluated as a failure condition in this project. A recalibration of the blur threshold against a manual sample of blurred frames would be needed as future work before drawing conclusions.

### Rapid Camera Movement

| Rapid camera | Heuristic mean | Heuristic median | MiDaS mean | MiDaS median | Frame count |
|---|---:|---:|---:|---:|---:|
| False | 45.62 | 28.92 | 97.91 | 81.45 | 11,819 |
| True | 99.96 | 68.67 | 161.56 | 132.47 | 290 |

Rapid camera movement was a significant failure condition for **both** methods, but affected them to different degrees. Heuristic error more than doubled under this condition (+119%, 45.62 → 99.96), a proportionally larger degradation than MiDaS's (+65%, 97.91 → 161.56). In absolute terms, however, heuristic under rapid camera movement (99.96) still outperformed MiDaS under normal conditions (97.91). This shows that rapid camera movement narrows heuristic's advantage over MiDaS without eliminating it. 

This suggests both methods share a common vulnerability tied to camera motion, possibly because the heuristic's bounding-box-height signal and MiDaS's depth estimate are both disrupted by the same underlying cause (fast pixel motion degrading detection/localization quality generally). Frequent track ID switches were also observed to occur under rapid camera movement during manual track labeling, though this was not systematically quantified. Future work might consist of evaluating track vs no-track performance under this condition.

<div class="image-carousel">

  <img src="../../assets/images/match_1_round_1_frame000257_rapid_camera_worst.png" alt="Rapid camera worst, match 1 round 1">
  <img src="../../assets/images/match_1_round_2_frame000217_rapid_camera_low_error_despite_condition.png" alt="Rapid camera low error, match 1 round 2">
  <img src="../../assets/images/match_2_round_1_frame000756_rapid_camera_worst.png" alt="Rapid camera worst, match 2 round 1">
  <img src="../../assets/images/match_3_round_2_frame000618_rapid_camera_low_error_despite_condition.png" alt="Rapid camera low error, match 3 round 2">
  <img src="../../assets/images/match_3_round_2_frame004170_rapid_camera_worst.png" alt="Rapid camera worst, match 3 round 2">

</div>

## Limitations
- **Dataset size.** Only 9 of 64 available rounds (6 train / 3 validation) were used, due to hardware constraints> Both frame extraction and MiDaS depth estimation are computationally expensive. Results may not generalize to different maps, matches, or player rosters beyond what was evaluated.
- **BEV scale instability.** Both BEV methods require empirical per-video scale recalibration against manually labeled ground truth. Scale is not a fixed, predictable constant under either method.
- **MiDaS frame-to-frame noise.** As discussed above, MiDaS-based trajectories exceed physically plausible player movement, and this was not resolved by normalization tuning. Thus suggesting a more fundamental limitation of applying a general-purpose monocular depth model frame-independently, without any temporal smoothing or consistency constraint. Although, due to harware limitations, only the medium-sized MiDaS DPT-Hybrid was used. The larger MiDaS model might yield better performance.
- **Manual track labeling dependency.** Both calibration and tracked evaluation depend on manually labeled `track_id` to `player_name` mappings. Tracks with ambiguous or switched identity (a `track_id` assigned to more than one real player during a round) were excluded calibration and were matched using the Hungarian algorithm for evaluation in an attempt to resolve the issue. Manual labelling are inherently subject to human error.
- **Empirical rather than analytical camera calibration.** The rotation convention mapping camera-relative BEV coordinates to world coordinates was determined by testing four candidate conventions against labeled calibration data and selecting the best fit, rather than derived analytically from a known camera model / FOV specification. This is common in real-life scenarios where camera specs are not known and assumes the winning convention generalizes correctly across all videos, which was not independently verified beyond the calibration set and might introduce error.
- **POV/camera selection heuristic.** The recorded camera each round is the player who survived longest (`survival_ratio`), used synonymously for "most likely spectated." This is a reasonable heuristic but does not guarantee the recorded viewpoint had good visual coverage of all other players.
- **Synchronization assumes constant frame-to-tick mapping.** Video-to-demo synchronization uses linear interpolation between a round's start and end, which assumes a constant, drift-free relationship between video time and game time across the whole round. Manual video capture might also introduce noise to the synchronization process.
- **No-track evaluation asymmetry.** As noted above, the untracked baseline lacks the manual exclusion filtering (POV-self, dead bodies) applied to the tracked evaluation.
- **Hardware constraints.** All experiments were run on an Apple M3 Pro (11-core: 5 performance + 6 efficiency, 18GB memory), using PyTorch's MPS (Apple Metal) backend as no CUDA-capable GPU was available. This limited the MiDaS model variant used (DPT-Hybrid rather than a larger variant), dataset size, and the scope of parameter tuning performed (e.g. the MiDaS grid search was restricted to the calibration subset rather than the full dataset).

## Design Decisions
**Monocular RGB input.** The pipeline was deliberately constrained to standard broadcast-style first-person video, rather than requiring specialized tracking hardware or multi-camera setups. These videos resembles the actual data esports broadcasts already produce, making the approach applicable without additional capture infrastructure. This also allows the pipeline to be applied to data beyong the context of esports.

**YOLO26m for detection.** A single-stage detector was chosen over two-stage alternatives specifically for the efficiency of single-stage architectures when processing video at scale. Detection would run once per frame across a full match, and per-frame detection latency was a first-order concern given hardware constraints. Only the **person** class is detected, since player localization was the sole objective and other classes were not pertinent to evaluation.

**ByteTrack for tracking.** ByteTrack showed great improvement from last tracker used for the tracking process (see [log](../log.md#week-5)) due to its usage of both high and low confidence detections. ByteTrack also uses motion and bounding-box continuity rather than appearance embeddings to associate detections, which is excellent for this project where all players on the same team are visually near-identical. Other models relying appearance-based re-identification showed significant reductiion in performance.

**Two depth methods compared.** Rather than committing to a single depth-estimation strategy, the pipeline was built to support both a cheap, interpretable geometric heuristic and a general-purpose learned model (MiDaS), specifically to allow direct empirical comparison of an accuracy/interpretability/compute tradeoff rather than assuming one approach's superiority.

**Device configuration.** For both detection and MiDaS depth extraction proccess which works directly with RGB frames and have more computational costs, the device configuration auto-detects and prioritizes using GPU over MPS (Apple Metal) over CPU to save computation time for these time consuming processes.

**Resumable, config-driven pipeline architecture.** Every pipeline stage checks for existing outputs per video before running (using `.done` markers for directory-based stages, output-file existence for tabular stages) and skips work that is already complete. This can eliminate the cost for losing progress to a crash partway through a long-running stage like MiDaS depth extraction. Per-step configuration is stored as separate, independently tunable YAML files merged onto validated dataclass schemas at runtime (using <a href="https://omegaconf.readthedocs.io/en/2.3_branch/" target="_blank" rel="noopener">OmegaConf</a>). It is a more modular design and allows for typos or invalid values in a tuned config to be caught immediately rather than silently ignored or propagating into results.

**Calibration-set-based evaluation design.** Because camera-relative BEV predictions cannot be compared to absolute ground-truth positions without first solving for an unknown rotation convention and scale factor, a subset of manually labeled, non-ambiguous predictions was set aside specifically to solve for these two unknowns using least-squares fitting, before evaluating on the full dataset. Relative spatial accuracy was included specifically because it is invariant to a uniform miscalibration in this transform, providing a check on whether the pipeline correctly captures player formation even when absolute placement calibration is imperfect. Trajectory consistency was included as the only metric requiring no ground truth at all, since it directly measures physical plausibility (frame-to-frame displacement against a known movement-speed ceiling) independent of whether calibration or matching is even correct.
