# Discussion

This section interprets the results from experiments, examines specific failure conditions, and discusses the limitations and design rationale behind the pipeline.

## Result interpretation

**Heuristic vs. MiDaS.** 

The heuristic method outperformed MiDaS across every metric and every individual video by a large, uniform gap (2.1x lower Euclidean error, 1.8x lower relative spatial error, 4.5x lower trajectory step distance). Some evidence might suggest why that is the case.

First, **MiDaS's trajectory step distance (20.03 units/frame) is well above what a real player can physically achieve** in a single frame. Assuming standard CS2 running speed (~250 units/second) and a ~30fps extraction rate, plausible per-frame movement is capped around 8–9 units/frame. Heuristic's step distance 4.42 is reasonably within this range, while MiDaS's 20.03 is not. This indicates MiDaS-based trajectories are not primarily tracking real player movement. This elevated number likely comes from frame-to-frame estimation noise, most plausibly from MiDaS producing slightly different depth values for the same physical scene between adjacent frames.

Second, **parameter tuning aimed directly at MiDaS's normalization (`depth_window`, `low_pct`, `high_pct`) produced almost no improvement** (calibration error 129.50 → 129.53), despite a grid search over a meaningfully wide parameter range. This is evidence that the error is not a tunable calibration/scaling problem, and further supports that the error is primarily irreducible (at least through normalization) per-frame depth-estimation noise.

**Scale instability** was also more severe for MiDaS than heuristic. Computing the spread of per-video scale factors from the [results table](results.md#heuristic-vs-midas): heuristic ranged from 4.311–6.176 (a 43% spread relative to its minimum), while MiDaS ranged from 0.899–1.570 (a 75% spread). Both methods require per-video scale recalibration, but MiDaS's substantially wider variance is a sign of having the same underlying instability rather than a separate issue.

All together, these results support a specific claim:

!!! tip ""
    For this task, a simple geometric heuristic approach based on bounding-box size is a more reliable depth extraction method than a general-purpose monocular depth model, primarily because of the model's frame-to-frame instability.

**Track vs. no track.**

!!! note "Placeholder"
    This discussion should be completed once numeric results from the tracking ablation are available. Expected discussion points once data exists: whether tracking's benefit is primarily in positional accuracy (Euclidean/relative spatial error) or in enabling continuity-dependent outputs (trajectories) that are undefined without it; and how the evaluation asymmetry noted above (missing exclusion filtering in the no-track condition) should be weighed when interpreting the size of the gap.

## Failure Cases
Three challenging conditions were flagged programmatically as a part of the research for this project:

- **Occlusion**: frames where two or more detected bounding boxes overlap above an IoU threshold (0.3), indicating players close together or partially blocking one another.
- **Motion blur**: frames with low Laplacian variance (below 100.0) in the extracted image, indicating fast camera or subject motion blurring detail.
- **Rapid camera movement**: frames where the synchronized camera yaw changed by more than 5° from the previous frame (wraparound-corrected at the ±180° boundary), indicating a fast pan/turn.

!!! note "Rapid camera movement"
    Camera yaw threshold (5°) might be different if original video was extracted at another frame rate.

### Occlusion

!!! note "Placeholder — needs results"
    Fraction of frames flagged and mean error with/without the condition (from `analyze_failure_modes.py` output) to be inserted, along with 2–3 annotated qualitative example frames.

### Motion Blur

!!! note "Placeholder — needs results"
    Same as above.

### Rapid Camera Movement

!!! note "Placeholder — needs results"
    Same as above. Note for interpretation once data is available: this condition is expected to interact with the heuristic-vs-MiDaS finding above, since rapid camera movement plausibly correlates with motion blur and could compound depth-estimation noise for MiDaS specifically — worth checking whether MiDaS's error gap versus heuristic widens specifically under this condition.

## Limitations
- **Dataset size.** Only 9 of 64 available rounds (6 train / 3 validation) were used, due to hardware constraints> Both frame extraction and MiDaS depth estimation are computationally expensive. Results may not generalize to different maps, matches, or player rosters beyond what was evaluated.
- **BEV scale instability.** Both BEV methods require empirical per-video scale recalibration against manually labeled ground truth. Scale is not a fixed, predictable constant under either method, and is notably worse for MiDaS (see [comparison](#result-interpretation)).
- **MiDaS frame-to-frame noise.** As discussed above, MiDaS-based trajectories exceed physically plausible player movement, and this was not resolved by normalization tuning. Thus suggesting a more fundamental limitation of applying a general-purpose monocular depth model frame-independently, without any temporal smoothing or consistency constraint. Although, due to harware limitations, only the medium-sized MiDaS DPT-Hybrid was used. The larger MiDaS model might yield better performance.
- **Manual track labeling dependency.** Both calibration and tracked evaluation depend on manually labeled `track_id` to `player_name` mappings. Tracks with ambiguous or switched identity (a `track_id` assigned to more than one real player during a round) were excluded calibration and were matched using the Hungarian algorithm for evaluation in an attempt to resolve the issue. Manual labelling are inherently subject to human error.
- **Empirical rather than analytical camera calibration.** The rotation convention mapping camera-relative BEV coordinates to world coordinates was determined by testing four candidate conventions against labeled calibration data and selecting the best fit, rather than derived analytically from a known camera model / FOV specification. This is common in real-life scenarios where camera specs are not known and assumes the winning convention generalizes correctly across all videos, which was not independently verified beyond the calibration set and might introduce error.
- **POV/camera selection heuristic.** The recorded camera each round is the player who survived longest (`survival_ratio`), used synonymously for "most likely spectated." This is a reasonable heuristic but does not guarantee the recorded viewpoint had good visual coverage of all other players.
- **Synchronization assumes constant frame-to-tick mapping.** Video-to-demo synchronization uses linear interpolation between a round's start and end, which assumes a constant, drift-free relationship between video time and game time across the whole round. Manual video capture might also introduce noise to the synchronization process.
- **No-track evaluation asymmetry.** As noted above, the untracked baseline lacks the manual exclusion filtering (POV-self, dead bodies) applied to the tracked evaluation.
- **Hardware constraints.** *[GPU/CPU specification — placeholder]* limited model used, dataset size, and the scope of parameter tuning performed (e.g. MiDaS grid search was restricted to the calibration subset rather than the full dataset).

## Design Decisions
**Monocular RGB input.** The pipeline was deliberately constrained to standard broadcast-style first-person video, rather than requiring specialized tracking hardware or multi-camera setups. These videos resembles the actual data esports broadcasts already produce, making the approach applicable without additional capture infrastructure. This also allows the pipeline to be applied to data beyong the context of esports.

**YOLO26m for detection.** A single-stage detector was chosen over two-stage alternatives specifically for the efficiency of single-stage architectures when processing video at scale. Detection would run once per frame across a full match, and per-frame detection latency was a first-order concern given hardware constraints. Only the **person** class is detected, since player localization was the sole objective and other classes were not pertinent to evaluation.

**ByteTrack for tracking.** ByteTrack showed great improvement from last tracker used for the tracking process (see [log](../log.md#week-5)) due to its usage of both high and low confidence detections. ByteTrack also uses motion and bounding-box continuity rather than appearance embeddings to associate detections, which is excellent for this project where all players on the same team are visually near-identical. Other models relying appearance-based re-identification showed significant reductiion in performance.

**Two depth methods compared.** Rather than committing to a single depth-estimation strategy, the pipeline was built to support both a cheap, interpretable geometric heuristic and a general-purpose learned model (MiDaS), specifically to allow direct empirical comparison of an accuracy/interpretability/compute tradeoff rather than assuming one approach's superiority.

**Device configuration.** For both detection and MiDaS depth extraction proccess which works directly with RGB frames and have more computational costs, the device configuration auto-detects and prioritizes using GPU over MPS (Apple Metal) over CPU to save computation time for these time consuming processes.

**Resumable, config-driven pipeline architecture.** Every pipeline stage checks for existing outputs per video before running (using `.done` markers for directory-based stages, output-file existence for tabular stages) and skips work that is already complete. This can eliminate the cost for losing progress to a crash partway through a long-running stage like MiDaS depth extraction. Per-step configuration is stored as separate, independently tunable YAML files merged onto validated dataclass schemas at runtime (using <a href="https://omegaconf.readthedocs.io/en/2.3_branch/" target="_blank" rel="noopener">OmegaConf</a>). It is a more modular design and allows for typos or invalid values in a tuned config to be caught immediately rather than silently ignored or propagating into results.

**Calibration-set-based evaluation design.** Because camera-relative BEV predictions cannot be compared to absolute ground-truth positions without first solving for an unknown rotation convention and scale factor, a subset of manually labeled, non-ambiguous predictions was set aside specifically to solve for these two unknowns using least-squares fitting, before evaluating on the full dataset. Relative spatial accuracy was included specifically because it is invariant to a uniform miscalibration in this transform, providing a check on whether the pipeline correctly captures player formation even when absolute placement calibration is imperfect. Trajectory consistency was included as the only metric requiring no ground truth at all, since it directly measures physical plausibility (frame-to-frame displacement against a known movement-speed ceiling) independent of whether calibration or matching is even correct.