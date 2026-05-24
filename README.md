# Monocular BEV Localization in FPS

This project explores monocular bird’s-eye-view (BEV) localization of players in first-person gameplay using computer vision techniques.

## Structure
- src/: core pipeline code
- scripts/: runnable scripts
- data/: dataset (not tracked)
- outputs/: results and visualizations

## Setup
pip install -r requirements.txt
pip freeze > requirements-lock.txt
source venv/bin/activate
python3 -m scripts.file_name

## Status
Week 1: Data pipeline setup
Week 2: Integrate YOLOv8 for detection