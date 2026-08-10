# Overview

This project investigates the following questions:

- How accurately can player positions be approximated in a <abbr title="Bird’s-Eye View">BEV</abbr> representation using only monocular RGB video input (no game engine information)?
- How do simple geometric heuristics compare to learned monocular depth estimation models for BEV reconstruction using first-person footage?
- To what extent does temporal tracking improve the stability and accuracy of spatial estimates?

The pipeline combines object detection, multi-object tracking, and two depth estimation approaches to generate camera-relative BEV positions. Engine-level demo files are used exclusively as ground truth for evaluation, allowing the system to be assessed under controlled conditions while remaining applicable to ordinary gameplay (or other) footage.

The project emphasizes modularity and reproducibility, allowing individual pipeline components to be replaced, tuned, and evaluated independently. Intermediate outputs are stored in standardized formats, making the framework suitable for experimentation, benchmarking, and future extensions.

## Motivation
Estimating player locations from a single RGB camera remains a difficult problem because depth information is not directly observable. Existing localization approaches often require multiple calibrated cameras, specialized sensors, or direct access to engine data, all of which limit their applicability to real gameplay recordings.

Counter-Strike 2 provides a unique opportunity to investigate this problem. Professional matches are publicly available together with engine-level demo files that record the complete game state at every simulation tick. These demo files provide highly accurate player positions and camera information that can be used to generate ground truth while allowing the localization pipeline itself to operate exclusively on recorded gameplay videos.

## Project Objectives
The primary objectives of this project are:

* Develop a complete modular pipeline for monocular bird's-eye-view player localization.
* Recover player trajectories using only RGB gameplay footage during inference.
* Compare heuristic and learned monocular depth estimation methods within the same localization framework.
* Evaluate each pipeline stage independently through quantitative experiments and visual analysis.
* Produce a reproducible research pipeline with configurable components and standardized intermediate outputs.

## Key Features

* **Modular pipeline architecture** allowing individual stages to be replaced or extended independently.
* **Automatic synchronization** between recorded gameplay videos and engine-level demo data.
* **Two interchangeable depth estimation approaches** for direct comparison.
* **Standardized intermediate datasets** stored as Parquet files for reproducibility and debugging.
* **Comprehensive visualization tools** for detections, tracking, trajectories, depth estimation, and BEV localization.
* **Configurable evaluation framework** supporting quantitative metrics and qualitative comparisons.
* **Interactive documentation** describing every stage of the pipeline and design decisions.

## Pipeline Overview

```pipeline
direction: TB

I[Input]
---
tooltip: Input
color: grey
type: io
---

A[Detection]
---
link: report/methodology/#detection
tooltip: Identifies player locations from individual RGB frames using an object detector.
color: blue
---

B[Tracking]
---
link: report/methodology/#tracking
tooltip: Associates detections across frames to construct continuous player trajectories.
color: orange
---

C[Heuristic Depth]
---
link: report/methodology/#heuristic
tooltip: Estimates depth using geometric assumptions based on player bounding box size.
color: green
---

D[MiDaS Depth]
---
link: report/methodology/#midas
tooltip: Uses monocular depth estimation to infer scene depth from RGB frames.
color: red
---

E[BEV Estimation]
---
link: report/methodology/#bev-estimation
tooltip: Transforms detections and estimated depth into camera-relative bird's-eye-view coordinates.
color: purple
---

T[Trajectory Construction]
---
link: report/methodology/#bev-estimation
tooltip: Aggregates per-frame BEV positions into per-track trajectories over time.
color: wine
---

O[Output]
---
tooltip: Output
color: grey
type: io
---

F[Evaluation]
---
link: report/methodology/#evaluation-metrics
tooltip: Compare predicted positions against ground truth.
color: teal
type: decision
---

I --> A
A --> B
B --> C
B --> D
C --> E
D --> E
E --> T
T --> O
O --> F
```

The localization pipeline is composed of five independent stages.

1. **Detection** identifies player locations from individual RGB frames.
2. **Tracking** associates detections across consecutive frames to construct player trajectories.
3. **Depth Estimation** estimates camera-relative player distance using either a geometric heuristic or the MiDaS monocular depth model.
4. **BEV Estimation** projects tracked detections into camera-relative bird's-eye-view coordinates.
5. **Trajectory Construction** builds BEV estimations into tracked player trajectories.
6. **Evaluation** compares predicted positions against synchronized ground truth generated from CS2 demo files.

Each stage produces standardized outputs that serve as inputs to the following stage, enabling components to be evaluated independently and replaced without modifying the remainder of the pipeline.

Click any stage in the diagram above to explore its implementation and methodology.

## Sample Outputs
The pipeline produces intermediate outputs for every stage in addition to the final localization results.

<div class="image-carousel">

  <video controls data-caption="Detection, match 1 round 2">
    <source src="../assets/visualization_gallery/detection/match_1_round_2.mp4" type="video/mp4">
  </video>
  <video controls data-caption="Tracking, match 1 round 2">
    <source src="../assets/visualization_gallery/tracking/match_1_round_2.mp4" type="video/mp4">
  </video>
  <img src="../assets/visualization_gallery/depths/match_1_round_2.png" alt="Depth map extraction, match 1 round 2">
  <video controls data-caption="BEV estimation (Heuristic), match 1 round 2">
    <source src="../assets/visualization_gallery/bev/heuristic/match_1_round_2.mp4" type="video/mp4">
  </video>
  <video controls data-caption="BEV estimation (MiDaS), match 1 round 2">
    <source src="../assets/visualization_gallery/bev/midas/match_1_round_2.mp4" type="video/mp4">
  </video>
  <video controls data-caption="Combined, match 1 round 2">
    <source src="../assets/visualization_gallery/combined/match_1_round_2.mp4" type="video/mp4">
  </video>
  <video controls data-caption="BEV vs GT (Heuristic), match 1 round 2">
    <source src="../assets/visualization_gallery/bev_vs_gt/match_1_round_2_heuristic_bev_vs_gt.mp4" type="video/mp4">
  </video>
  <video controls data-caption="BEV vs GT (MiDaS), match 1 round 2">
    <source src="../assets/visualization_gallery/bev_vs_gt/match_1_round_2_midas_bev_vs_gt.mp4" type="video/mp4">
  </video>

</div>

Each visualization is generated directly from the pipeline.

## Acknowledgements
This project was completed as part of an undergraduate research project for CSCD94 at the University of Toronto Scarborough.

Special thanks to my supervisor <a href="https://www.utsc.utoronto.ca/cms/francisco-estrada" target="_blank" rel="noopener">Francisco Estrada</a> for guidance throughout the project and to the developers and maintainers of the open-source libraries that made this work possible: 

- <a href="https://opencv.org/" target="_blank" rel="noopener">OpenCV</a>
- <a href="https://www.ultralytics.com/" target="_blank" rel="noopener">Ultralytics YOLO</a>
- <a href="https://github.com/roboflow/trackers" target="_blank" rel="noopener">Roboflow Trackers</a> (ByteTrack implementation)
- <a href="https://supervision.roboflow.com/" target="_blank" rel="noopener">Roboflow Supervision</a>
- <a href="https://huggingface.co/Intel/dpt-hybrid-midas" target="_blank" rel="noopener">Intel DPT-Hybrid (MiDaS) via Hugging Face Transformers</a>
- <a href="https://github.com/LaihoE/demoparser" target="_blank" rel="noopener">demoparser2</a>

Gameplay data was collected from publicly available professional <a href="https://www.counter-strike.net/cs2" target="_blank" rel="noopener">Counter-Strike 2</a> matches distributed through <a href="https://www.hltv.org/" target="_blank" rel="noopener">HLTV</a>.
