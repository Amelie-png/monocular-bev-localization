# Monocular BEV Localization

Undergraduate research project for CSCD94 on monocular bird's-eye-view (BEV) localization of players in first-person gameplay footage.

🔗 **Live site:** https://amelie-png.github.io/monocular-bev-localization/

## Overview

This repo contains the code, experiments, and report for a pipeline that detects, tracks, and localizes players in a top-down BEV representation using only monocular (single-camera) input from gameplay video.

## Contents

- src/: modular core pipeline code
- scripts/: runnable scripts
- data/: dataset
- outputs/: results and visualizations
- explorations/: exploratory code -- not used in pipeline
- docs/ and overrides/: code for website -- not used in pipeline

## Setup

### Install requirements
pip install -r requirements.txt
### Freeze current requirements for version control
pip freeze > requirements-lock.txt
### Activate virtual environment
source venv/bin/activate

## Run pipeline

### To run any script files use the following command
python3 -m scripts.file_name

### Data parsing
- parse_demo: Parse downloaded .demo files
- extract_frames: Extract frames and frame information from downloaded .mp4 (video) files
- sync_data: Sync demo data and video data
If needed:
- reset_data: Reset data/processed and outputs/ folder

### Detection
- tune_detection: Run detection with different configuration of model and parameter
- Manually inspect tuning_summary.csv and select few configs with best metrics
- evaluate_detection: Evaluate detection and produce visualization with given config
- Manually inspect output visualization video and select config with best results
- Evaluate detection on validation dataset
- run_detection: Produce detection files for next steps in pipeline

### Tracking
- tune_tracking: Run tracking with different configuration of parameter
- Manually inspect tuning_summary.csv and select few configs with best metrics
- evaluate_tracking: Evaluate tracking and produce visualization with given config
- Manually inspect output visualization video and select config with best results
- Evaluate tracking on validation dataset
- run_tracking: Produce tracking files for next steps in pipeline

## Author

Amelie Zhang — University of Toronto Scarborough
