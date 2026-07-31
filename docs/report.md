# Project

## Data collection
### Dataset sources
### Demo parsing
### Frame extraction
### Ground truth
### Data preprocessing

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

![Result 1](assets/result_1.gif)
![Result 2](assets/result_2.png)
![Result 3](assets/result_3.png)

</div>

## Discussion
### Why certain methods performed better
### Failure cases
### Limitations
### Design decisions

<div class="split" markdown>

<div class="split__text" markdown>

### Detection

We use YOLOv? fine-tuned on annotated gameplay frames to detect player
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