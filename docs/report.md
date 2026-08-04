# Project Report

---

## Overview
This project investigates a challenging form of spatial reasoning under limited visual input, where depth and scale must be inferred rather than directly measured. By using FPS gameplay as a controlled yet dynamic testbed, the work contributes to understanding how computer vision systems perform in scenarios analogous to real-world applications such as surveillance and sports analytics, where only monocular video is available.

We ask of the follow questions:

* How accurately can player positions be approximated in a bird’s-eye-view representation using only monocular video input?
* How do simple geometric heuristics compare to learned monocular depth estimation models for BEV reconstruction in FPS environments?
* To what extent does temporal tracking improve the stability and accuracy of spatial estimates?

---

## Data collection
Data collection is an imperative step to any research question, and oftentimes, the most difficult step. To overcome the challenge of procuring data for training machine learning models and pipelines, the proposal is to use video game data to mimic real life data. The video game chosen for data collection is <a href="https://www.counter-strike.net/cs2" target="_blank" rel="noopener">Counter-Strike 2 (CS2)</a> for its publicly available data and realistic artstyle. However, there are [limitations](#limitations) to this data.

<div class="grid cards" markdown>

- **HLTV Demo Files**

    Downloads from HLTV.

    [:octicons-arrow-right-24: Demo Parsing](#demo-parsing)

- **Demo Parsing**

    Generates structured data.

    [:octicons-arrow-right-24: Frame Extraction](#frame-extraction)

- **Ground Truth**

    Synchronizes frames with demo ticks.

</div>

### Dataset sources
The dataset needed for this project can be sourced from CS2 pro games available through <a href="https://www.hltv.org/" target="_blank" rel="noopener">HLTV</a>. In professional CS2 tournaments, a series is often played as a best-of-three. Each map within the series produces a separate demo file, resulting in two or three demo files depending on the outcome of the game. Each demo file contains the comprehensive engine state logged per tick in a match. This is the most central part of the dataset. For this project specifically, data was collected from the matches played on <a href="https://www.hltv.org/matches/2394174/natus-vincere-vs-vitality-iem-atlanta-2026" target="_blank" rel="noopener">May 15, 2026 Natus Vincere vs. Vitaly game</a>. The gameplay videos are screen recordings of in game demo replay. These RGB videos are inputs into the pipeline and the demo files are used only to generate ground truth for evaluation.

### Demo parsing
Once the demo files are downloaded and extracted, they need to be parsed into readable and manipulable data. This project uses tools from the <a href="https://github.com/LaihoE/demoparser" target="_blank" rel="noopener">demoparser2</a> library and extracts player positions, player identifiers, camera information, round metadata, timing information, and other fields required throughout the pipeline. Parquet format was chosen for its efficient columnar storage and fast loading of large structured datasets and is also the chosen format for all data files further down the pipeline.

!!! example "Example: `player_positions.parquet`"
    *(Table placeholder)*

Demo parsing also produces a recording_plan file which outlines the necessary information to screen record the rounds of a match in game for the rest of data collection. Most importantly, it chooses the player <abbr title="Point-of-View">POV</abbr> and position which will become the camera by choosing the player with the largest survival ratio, minimising camera switches and maximising continuous observations. 

!!! note
    `recording_plan` is required during data collection to guide replay recording. Demo files are retained for producing ground truth for evaluation only.

!!! example "Example: `recording_plans.parquet`"
    *(Table placeholder)*

### Frame extraction
The recorded replay videos can be accessed in game with the corresponding demo file (tutorials for replay recording are widely available online). The replay recordings for this project are recorded per round in a match. The start and end of a round, duration of a round, as well as the player POV needed for a complete video capture are outlined in the recording plan for the corresponding match. The recordings are separated into rounds for a controlled camera POV and have a fixed buffer at the beginning of each recording to facilitate video parsing.

(video example of recorded rounds)

Frame extraction is done using the <a href="https://pypi.org/project/opencv-python/" target="_blank" rel="noopener">cv2</a> library at the video's native frame rate to preserve temporal continuity for multi-object tracking but can be done at specified <abbr title="Frames Per Second">fps</abbr> of choice. The frame extraction process saves each extracted frame as a `.png` file and stores frame metadata in Parquet format. Frame metadata is used further down the pipeline and is important data for evaluation.

!!! example "Example: `frame_metadata.parquet`"
    *(Table placeholder)*

### Ground truth
To produce the ground truth needed for evaluation of the pipeline, the first step is to sync the demo data to the recorded videos as they are measured in different units. The `RoundSynchronizer` module uses linear interpolation to match a frame from a video to a tick in a demo file, knowing the start and end timestamp of a video and the start and end tick of a round and while assuming constant mapping between the two. The sync data is stored in `.parquet` files.

!!! example "Example: `sync.parquet`"
    *(Table placeholder)*

Now the frames are matched to a tick in the demo file. However, the <abbr title="Bird’s-Eye View">BEV</abbr> estimations produced by the pipeline are relative to the camera and not in coordinate format like the X, Y, Z in demo files. To convert camera-relative predictions, the camera position (same as position of player_pov) and the camera rotation will be needed, both obtained from the syncing process. Rotation convention and scale factor for the conversion are calibrated independently for each round because these values differ between rounds.

!!! example "Example: `coordinates.parquet`"
    *(Table placeholder)*

To match the ground truth to the actual tracked players for evaluation, this project uses 2 possible approaches. For track IDs with clear manual labels to a certain player, simply join the track ID to the player name to the corresponding ground truth. Otherwise for pseudo-tracks (fake tracks for no track evaluation of the pipeline) and ambiguous labels where a track ID is assigned to different players at different instances in the round or when manual labels are not available, `match_frame` uses the Hungarian algorithm to assign track IDs to players such that it minimizes total distance across all pairs of predicted points and ground truth points simultaneously and not just greedily picking the nearest for each point. The predicted points and the corresponding player data are stored in `.parquet` files for evaluation.

!!! example "Example: `track_label.parquet`"
    *(Table placeholder)*

### Data preprocessing
Although frame extraction supports optional image preprocessing operations, such as cropping to remove <abbr title="User Interface">UI</abbr> elements or other irrelevant regions, no preprocessing is applied to the datasets used in this project. All experiments reported in this work are conducted using the original extracted frames.

(image example of original vs cropped frame)

---

<div class="grid cards" markdown>

-   :material-magnify: **Detection**

    ---

    YOLO-based player detection on raw gameplay frames.

-   :material-motion-outline: **Tracking**

    ---

    Multi-object tracking across frames with ByteTrack.

-   :material-map-outline: **BEV Localization**

    ---

    Projects tracked players into a top-down map view.

</div>

## Methodology
### Overall pipeline
### Detection
### Tracking
### BEV localization
### Synchronization
### Evaluation metrics

## Results
### Quantitative metrics
### Visualizations
### Example outputs
### Comparisons

<div class="gallery" markdown>

![Result 1](assets/placeholder.png)
![Result 2](assets/placeholder.png)
![Result 3](assets/placeholder.png)

</div>

## Discussion
### Why certain methods performed better
### Failure cases
### Limitations
### Design decisions

<div class="split" markdown>

<div class="split__text" markdown>

### Detection

Why use YOLOv? fine-tuned on annotated gameplay frames to detect player
bounding boxes with high recall under motion blur and occlusion.

</div>

<div class="split__image" markdown>
![Detection example](assets/detection.png)
</div>

</div>

## Conclusion and Future Work
### What was achieved
### Research question revisited
### Better detectors
### Better tracking
### Learned BEV models
### Larger datasets
### Real-time inference