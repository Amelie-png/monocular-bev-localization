# Monocular BEV Localization in FPS

This project explores monocular bird’s-eye-view (BEV) localization of players in first-person gameplay using computer vision techniques.

## Structure
- src/: core pipeline code
- scripts/: runnable scripts
- data/: dataset (not tracked)
- outputs/: results and visualizations

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
- Manually inspect output visualization video and select congif with best results
- Evaluate detection on validation dataset
