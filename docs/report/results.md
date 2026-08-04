# Results

Three professional CS2 matches containing 64 recorded rounds were collected as the dataset. Due to hardware limitations, 9 rounds were randomly selected for the rest of the pipeline: 6 used as a training/calibration set and 3 as a validation set. Detection was performed using YOLO26m, tracking using ByteTrack, and BEV localization using both a geometric heuristic and a MiDaS-based depth approach.

## Detection
Player detection was performed using a YOLO26m model restricted to the **person** class, with predictions passed as input directly to the tracking stage.

A grid search was run over

    model_name ∈ { "yolo26n.pt", "yolo26m.pt" }
    confidence_threshold ∈ { 0.35, 0.45, 0.55 }
    
against the training set.

**Tuning summary**

(table)
(plot if possible)

From the configurations, X, Y, and Z had the best perfermance as they had (higher data, lower data, etc). They were selected to be visualized and evaluated manually. Specifically, the manual evaluation aim to maximize true detections and minimize missing players and false positives and to identify underlaying failure causes.

(example table for manual evaluation)

**Chosen configuration:**

| Parameter | Value |
|---|---|
| `model_name` | `yolo26m.pt` |
| `confidence_threshold` | 0.35 |
| `crop_bottom_ratio` | 0.0 |
| `batch_size` | 8 |

**Final detection visualization**

*[Detection visualization video]*

## Tracking
Multi-object tracking was performed using ByteTrack.

A grid search was run over

    track_activation_threshold ∈ { 0.25, 0.45, 0.65 }
    lost_track_buffer ∈ { 30, 60 }
    minimum_consecutive_frames ∈ { 2 }

against the training set.

**Tuning summary**

(table)
(plot if possible)

From the configurations, X, Y, and Z had the best perfermance as they had (higher data, lower data, etc). They were selected to be visualized and evaluated manually to maximize track continuation and minimize track switches.

(example table for manual evaluation)

**Chosen configuration:**

| Parameter | Value |
|---|---|
| `track_activation_threshold` | 0.25 |
| `lost_track_buffer` | 30 |
| `minimum_consecutive_frames` | 2 |

**Trajectory visualization**

Tracking visualization video is very similar to detection video. For clearer understanding of the tracking quality, the trajectory visualization is shown instead.

*[Trajectory visualization video]*

## BEV Estimation
Both BEV methods were tuned against a **calibration set** containing frames where manual track-to-player labeling gives trusted identity for a direct comparison between a projected prediction and its true ground-truth position. Tuning parameters were evaluated by their effect on mean [Euclidean positional error](methodology.md#evaluation-metrics) on this calibration set.

### Heuristic
The heuristic depth estimation method estimates player distance using geometric assumptions.

A grid search was run over

    fov_scale ∈ {0.7, 0.85, 1.0, 1.15, 1.3}

against the calibration set.

**Tuning Summary**

| Configuration | `fov_scale` | Mean calibration error |
|---|---|---|
| Default | 1.0 | 76.97 |
| Tuned | 1.15 | 75.24 |

The improvement is modest, and `fov_scale`'s selected value being close to 1.0 suggests the original ~90° FOV assumption was reasonably close to the camera's true FOV.

**Chosen configuration:**

| Parameter | Value |
|---|---|
| `fov_scale` | 1.15 |
| `depth_window` | 5 (default) |
| `low_pct` | 2.0 (default) |
| `high_pct` | 98.0 (default) |

### MiDaS
The MiDaS depth extraction method was performed using MiDaS DPT_Hybrid.

A grid search was run over

    depth_window ∈ {3, 5, 7, 9}
    low_pct ∈ {1, 2, 5}
    high_pct ∈ {95, 98, 99}
    fov_scale ∈ {0.7, 0.85, 1.0, 1.15, 1.3}

against the calibration set.
 
**Tuning Summary**
 
| Configuration | Mean calibration error |
|---|---|
| Default (`depth_window=5`, `low_pct=2`, `high_pct=98`, `fov_scale=1.0`) | 129.50 |
| Tuned | 129.53 |

**Chosen configuration:**
 
| Parameter | Value |
|---|---|
| `depth_window` | 5 |
| `low_pct` | 1.0 |
| `high_pct` | 95.0 |
| `fov_scale` | 1.15|

Despite the parameter search, MiDaS calibration error was practically unchanged (129.50 → 129.53). This result is discussed in [Discussion](discussion.md#result-interpretation). It shows that the MiDaS method's error is not primarily a calibration/normalization problem that can fixed with tuning.

## Comparisons

### Heuristic vs. MiDaS
Final per-video evaluation results, using each method's tuned configuration above:

| Video | Heuristic scale | Heuristic mean Euclidean error | MiDaS scale | MiDaS mean Euclidean error |
|---|---:|---:|---:|---:|
| match_1_round_1 | 5.333 | 62.49 | 1.224 | 85.01 |
| match_1_round_2 | 4.311 | 56.28 | 1.158 | 98.94 |
| match_2_round_1 | 5.163 | 49.85 | 1.570 | 113.41 |
| match_2_round_2 | 5.435 | 24.65 | 0.899 | 38.68 |
| match_3_round_1 | 6.176 | 55.71 | 1.496 | 100.40 |
| match_3_round_2 | 5.487 | 45.08 | 1.505 | 147.37 |

**Overall summary (all videos pooled)**

| Metric | Heuristic | MiDaS | Heuristic advantage |
|---|---:|---:|---:|
| Mean Euclidean positional error | 50.25 | 103.21 | ~2.1x lower |
| Mean relative spatial error | 36.05 | 65.07 | ~1.8x lower |
| Mean trajectory step distance | 4.42 | 20.03 | ~4.5x lower |

The heuristic method outperformed MiDaS across **all three** evaluation metrics, consistently, across every individual video. Full discussion and interpretation in [Discussion](discussion.md#result-interpretation).

*[Comparison plots (e.g. boxplots of Euclidean error per variant) — insert once `scripts/plot_results.py` output is available]*

*[Side-by-side comparison visualization (tracked video + heuristic BEV + MiDaS BEV) — insert once rendered]*

### Track vs. No Track
To isolate tracking's specific contribution to localization accuracy, a second BEV estimation was run using the heuristic method on **untracked** detections. As mentioned in [Data Collection](data_collection.md#ground-truth), a pseudo-tracking set was built by treating each frame's detections as independent (no persistent identity across frames). The pseudo-tracking set was matched to ground truth per-frame using Hungarian assignment rather than the manual track labels used elsewhere in evaluation. 

MiDaS was not evaluated in the no-tracking condition, since the heuristic-vs-MiDaS comparison above already established heuristic as the stronger BEV method, and the tracking ablation's purpose is to isolate tracking's effect rather than re-run every combination.

Trajectory consistency is not reported for the no-tracking condition, since it requires persistent identity across frames to measure, thus not applicable to the pseudo-tracking data. This is an expected limitation of the comparison, not a missing result.

!!! warning "Evaluation asymmetry"
    The no-tracking baseline does not apply the same POV-self / dead-body exclusion filtering used in the tracked evaluation, since that filtering relies on manually labeled persistent track IDs (not available for the pseudo-tracking set). The reported gap between tracked and untracked accuracy should therefore be read as an **upper bound** on tracking's benefit, not a perfectly isolated measurement.

| Metric | Heuristic (tracked) | Heuristic (no track) |
|---|---:|---:|
| Mean Euclidean error | 50.25 | *[fill in]* |
| Mean relative spatial error | 36.05 | *[fill in]* |
| Mean trajectory step distance | 4.42 | N/A (no persistent identity) |

*[Plots and visualization — insert once available]*

## Visualizations

<div class="image-carousel">

  <img src="/assets/hero.png" alt="Hero">
  <img src="/assets/favicon.png" alt="Favicon">
  <img src="/assets/placeholder.png" alt="Placeholder">

</div>