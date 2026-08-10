# Development Log

## Week 1
Project Initialization & Data Pipeline Setup

Objectives:

- Set up development environment (PyTorch, OpenCV, dependencies)
- Acquire and parse Counter-Strike 2 demo files with demoparser2
- Extract:
    - RGB frames (or recorded gameplay)
    - Ground truth player positions
    - Build synchronized data loader

Deliverable:

- Script that loads frames and aligned ground truth positions
- Initial dataset sample prepared

!!! warning "WIP"
    Ground truth and synchronization scripts written but needs further refinement

## Week 2
Detection Baseline

Objectives:

- Integrate pretrained YOLOv8
- Run inference on gameplay frames
- Visualize bounding boxes

Deliverable:

- Detection outputs visualized on sample clips
- Baseline detection pipeline functional

!!! failure "Blockers"
    Tried to use gameplay footage from official pro games streams, but had:
        
    - Bad video resolution
    - Camera cuts to non gameplay footage
    - No control over player POV
    - Frequent player POV switches in a round

    which negatively impacted detection

## Week 3
Detection Evaluation & Refinement

Objectives:

- Evaluate detection quality (precision/recall qualitatively)
- Identify common failure cases (missed players, false positives)

Deliverable:

- Clean, standardized detection outputs

!!! failure "Blockers"
    Bad performance with current YOLO models (YOLOv8)

!!! success "Improvements"
    Resolved recorded footage issue from last week, now screen capturing the footage from in-game replays yielding much better results

## Week 4
Multi-Object Tracking Integration

Objectives:

- Integrate DeepSORT
- Assign persistent IDs across frames
- Tune tracking parameters

Deliverable:

- Video with tracked players and consistent IDs

!!! failure "Blockers"
    Bad performance with DeepSORT, many false positives and tracks persisting beyond actual alive time

!!! success "Improvements"
    Resolved detection performance issues by switching to YOLO26 models

## Week 5
Tracking Stabilization & Validation

Objectives:

- Evaluate:
    - ID switches
    - Track fragmentation
- Implement smoothing (e.g., moving averages)
- Improve robustness under occlusion

Deliverable:

- Stable tracking outputs suitable for downstream use

!!! failure "Blockers"
    Due to time and resource (data) constraints, might not get to smoothing or improving robustness under occlusion

!!! success "Improvements"
    Resolved tracking performance issues by switching to ByteTrack model

## Week 6
Heuristic BEV Projection (Baseline)

Objectives:

- Implement geometric projection:
    - Bounding box scale → depth approximation
    - Flat ground plane assumption
- Map detections into 2D top-down space

Deliverable:

- First working BEV visualization (heuristic method)

!!! failure "Blockers"
    Wanted to implement top-down visualization onto minimap, but required ground truth and camera info, resorted to camera relative visualizations, might not get to minimap visualization due to time constraints

## Week 7
Depth-Based BEV Estimation

Objectives:

- Integrate MiDaS
- Generate depth maps
- Extract player depth and convert to BEV coordinates

Deliverable:

- BEV predictions using depth-based method

!!! failure "Blockers"
    Wanted to switch to higher performance MiDaS model but could not due to hardware constraints

## Week 8
Full Pipeline Integration

Objectives:

- Combine: Detection → Tracking → BEV (both methods)
- Ensure synchronization across all components
- Standardize output format (trajectories + predictions)

Deliverable:

- End-to-end pipeline producing:
    - Player trajectories
    - BEV predictions (heuristic + depth)

!!! warning "WIP"
    Needs to include visualization production in pipeline

!!! success "Improvements"
    Refined modularization of current code for pipeline building

## Week 9
Evaluation Framework

Objectives:

- Implement metrics:
    - Euclidean positional error
    - Trajectory consistency
    - Relative spatial accuracy
- Align predictions with ground truth

Deliverable:

- Evaluation scripts with initial outputs

!!! failure "Blockers"
    Needed a way to match track ID to ground truth as they are not inherently associated

!!! success "Improvements"
    Implemented metrics analysis for previous tuning stages in the same format as final evaluation

## Week 10
Core Experiments

Objectives:

- Run primary comparisons:
    - Heuristic vs depth-based BEV
    - With vs without tracking
- Collect quantitative results

Deliverable:

- Initial results tables and plots

!!! success "Improvements"
    Manual labelling track ID to player name (in ground truth) but still not accurate enough (e.g. when there are ID switches between players)

## Week 11
Extended Experiments & Ablations

Objectives:

- Analyze challenging conditions:
    - Occlusion
    - Motion blur
    - Rapid camera movement
- Identify failure modes

Deliverable:

- Expanded results + qualitative examples

!!! success "Improvements"
    Used Hungarian algorithm for ambiguously labelled tracks to improve ground truth to BEV estimation matching accuracy

## Week 12
Analysis & Final Report

Deliverable:

- Full report (digital) with visualizations
- Modular codebase with documentation

!!! success "Improvements"
    Added lots of visualization productiion for the final report