# Monocular BEV Player Localization

Estimate bird's-eye-view (BEV) player positions from monocular esports broadcast footage (Counter-Strike 2 POV recordings), using object detection, multi-object tracking, and two BEV estimation strategies: geometric heuristic vs MiDaS-based monocular depth approach. Predictions are evaluated against ground-truth player positions extracted directly from game demo files.

**Full project documentation, methodology, and results:** https://amelie-png.github.io/monocular-bev-localization/

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Try It: Sample Data](#try-it-sample-data)
- [Running the Pipeline](#running-the-pipeline)
- [Configuration](#configuration)
- [Repository Structure](#repository-structure)
- [Evaluation (Optional)](#evaluation-optional)
- [Troubleshooting](#troubleshooting)

## Overview

Given a match video and its corresponding `.dem` demo file, this pipeline produces:

- **Player detections** per frame (YOLO26)
- **Persistent player tracks** across frames (ByteTrack)
- **BEV position estimates** using either a geometric heuristic or a MiDaS-based depth model
- **Player trajectories** over time

Demo files are used **only** to generate ground truth for evaluation. The pipeline itself runs entirely on RGB video and requires no game-engine access at inference time. See [the full report](https://amelie-png.github.io/monocular-bev-localization/) for methodology, tuning results, and findings.

## Installation

**Requirements:** Python 3.11+, [ffmpeg](https://ffmpeg.org/) (system binary, required for combined visualizations).

A GPU (CUDA or Apple Metal/MPS) is recommended for detection and MiDaS depth estimation but not required. The pipeline falls back to CPU automatically.

```bash
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

`ffmpeg` isn't a Python package and must be installed separately:
```bash
brew install ffmpeg        # macOS
apt install ffmpeg         # Ubuntu/Debian
```

## Try It: Sample Data

The experiements in this project and the pipeline were run using footage and data from the [May 15, 2026 Natus Vincere vs. Vitality series](https://www.hltv.org/matches/2394174/natus-vincere-vs-vitality-iem-atlanta-2026).

The recorded footage and demo files are not included in this repository. Users who wish to reproduce the experiments should obtain the corresponding footage independently from its original source.

[More about data collection.](https://amelie-png.github.io/monocular-bev-localization/report/data_collection)

!!! note
    Match footage and demo files are are sourced from a publicly broadcast professional match. They are not included as they are not for distribution purposes. You can source your own videos following many tutorials online.

## Running the Pipeline

Run complete pipeline end-to-end:
```bash
python3 -m scripts.run_pipeline
```
By default this processes every video listed in `data/splits/train.txt`. Use `--video` to target specific videos, or edit the split file.

### Running individual stages

Each stage can also be run individually, which is useful for debugging, or if you only need to run part of the pipeline:

```bash
python3 -m scripts.data.extract_frames --video match_2_round_1
python3 -m scripts.detection.run_detection --video match_2_round_1
python3 -m scripts.tracking.run_tracking --video match_2_round_1
python3 -m scripts.bev.run_depth --video match_2_round_1
python3 -m scripts.bev.run_bev --video match_2_round_1          # heuristic (default)
python3 -m scripts.bev.run_bev --midas --video match_2_round_1  # MiDaS
```

Every pipeline stage is **resumable**: already-processed videos are automatically skipped. Pass `--force` to any script to recompute anyway.

### Visualizations

Visualizations are generated separately from the core pipeline:

```bash
python3 -m scripts.visualize_pipeline
```

This produces visualizations such as:

- Detection bounding boxes
- Tracking IDs
- BEV estimates
- Estimated trajectories
- Combined comparison videos

Visualization per stage is also available. Additional visualization requires ground truth that might not be available for independently sourced data. The visualizations are optional and are not required for the underlying pipeline outputs.

## Configuration

Per-step parameters live as YAML files under `configs/` (`detection.yaml`, `tracking.yaml`, `heuristic_bev.yaml`, `midas_bev.yaml`), merged onto validated schemas at runtime. An invalid value raises an error immediately rather than being silently ignored. Override individual values via CLI flags on any step script (e.g. `--confidence 0.35`), or edit the YAML directly.

## Repository Structure

`configs/` Per-step tuned configuration files
`data/`

- `raw/videos/` Input .mp4 files
- `raw/demos/` Input .dem files
- `splits/` Video list files (train.txt, sample.txt, ...)
- `processed/` All pipeline-generated data (frames, detections, tracks, bev, ...)
`outputs/`

- `videos/` All rendered videos (detection, tracking, bev, combined, ...)
- `images/` Depth samples, qualitative failure-mode examples
- `eval/` Evaluation results (parquet, results table)
- `plots/` Result comparison plots
- `tuning/` Parameter sweep results and configs
`scripts/` One script per pipeline step (CLI-runnable) and helper scripts organized into subfolders
`src/` Core implementation (detection, tracking, bev, depth, evaluation)

## Evaluation (Optional)

Evaluating predictions against ground truth is **not required to use the core pipeline**. 

It is a separate research layer used to produce the results in [the full report](https://amelie-png.github.io/monocular-bev-localization/report/). Evaluation requires a corresponding demo file and manually prepared evaluation data.

The evaluation workflow includes:

- Video/demo synchronization
- Ground-truth extraction
- Coordinate-system calibration
- Manual track labeling
- Metric computation
- Quantitative and qualitative analysis
- Estimation vs ground truth visualizations

See the [Data Collection](https://amelie-png.github.io/monocular-bev-localization/report/data_collection) and [Methodology](https://amelie-png.github.io/monocular-bev-localization/report/methodology) sections of the report for the complete evaluation workflow.


## Troubleshooting

- **A step does nothing when run**: it found existing output for every requested video and skipped it; pass `--force` to override.
- **`ffmpeg: command not found`**: install it separately (see [Installation](#installation)); required only for combined visualizations.
- **Missing depth files during `run_bev --midas`**: run `scripts.bev.run_depth` first.
- Check the printed statistics after each step (frame/detection/track counts), most data issues surface there first.
