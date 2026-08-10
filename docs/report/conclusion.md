# Conclusion
 
This project set out to answer the following questions:

- How accurately can player positions be approximated in a bird’s-eye-view representation using only monocular RGB video input?
- How do simple geometric heuristics compare to learned monocular depth estimation models for BEV reconstruction using first-person footage?
- To what extent does temporal tracking improve the stability and accuracy of spatial estimates?
 
A complete pipeline was built and evaluated end-to-end:

```pipeline
A[Detection]
---
link: ../methodology/#detection
tooltip: YOLO26m
color: green
---

B[Tracking]
---
link: ../methodology/#tracking
tooltip: ByteTrack
color: teal
---

C[Depth estimation]
---
link: ../methodology/#bev-localization
tooltip: Heuristic geometric and MiDaS-based
color: blue
---

D[BEV projection]
---
link: ../methodology/#bev-estimation
tooltip: BEV projection
color: purple
---

E[Trajectory construction]
---
link: ../results/#tracking
tooltip: Trajectory construction
color: wine
---

F[Evaluation]
---
link: ../methodology/#evaluation-metrics
tooltip: Evaluation
color: orange
---

A --> B
B --> C
C --> D
D --> E
D --> F
E --> F
```

with an evaluation framework aligning predictions to ground truth extracted directly from game demo files, using three complementary metrics (Euclidean positional error, relative spatial accuracy, trajectory consistency).
 
The central empirical finding was that **the simple geometric heuristic consistently and substantially outperformed the learned MiDaS-based approach** across all three metrics and all evaluated videos. Results suggested that it is not due to a fixable calibration issue (parameter tuning targeting MiDaS's normalization produced negligible improvement), but most likely due to frame-to-frame depth-estimation instability. This instability is likely caused by applying a general-purpose monocular depth model independently per frame, without temporal consistency constraints.
 
Tracking was also found to meaningfully improve BEV localization. When persistent identity was removed (i.e. no tracking), mean Euclidean error increased by roughly 53% (46.92 → 71.91) and mean relative spatial error by roughly 60% (33.18 → 53.06), with one round (`match_2_round_2`) showing a substantially larger effect (over 300%) than the rest. This suggests tracking's benefit, is not uniform across conditions, and matters most for rounds with closely-spaced players where per-frame identity matching is prone to error. This finding is discussed further, including a note about an evaluation asymmetry that means it should be read as an upper bound, in [Discussion](discussion.md#track-vs-no-track).
 
## Future Work

### Larger Datasets
Only 9 of 64 available rounds were evaluated due to hardware constraints. Expanding to the full round set, and to additional matches/maps or other video sources beyond this project's dataset, would both strengthen confidence in the heuristic-vs-MiDaS finding and test its generalization.
 
### Better Detectors
Fine-tuning YOLO26 on domain-specific labeled data (rather than a general person detector) could improve recall under occlusion and reduce sensitivity to non-standard player poses (crouching, prone) that a general-purpose detector may not represent well in its training distribution.

### Better Tracking
Resolving the ambiguous/ID-switched tracks with better tracking models and improving manual track ID label precision and accuracy to reliably produce usable ground truth. Since appearance-based re-identification is largely unavailable in this project, this likely requires stronger motion modeling or exploiting game-specific cues (e.g. team-side constraints) rather than visual appearance. Frequent track ID switches were also observed under rapid camera movement conditions (see [Discussion](discussion.md#rapid-camera-movement)). A tracker with stronger motion prediction during fast camera pans could specifically target this failure mode.
 
### Learned BEV Models
Improving depth extraction using general-purpose monocular depth model by using a newer and more <abbr title="State-Of-The-Art">SOTA</abbr> model. Additionally, incorporating temporal consistency directly into training, rather than treating each frame independently as a next step.
 
### Real-Time Inference
The current pipeline is fully offline/batch. MiDaS depth extraction in particular is the most computationally expensive stage per frame. Given the heuristic method's demonstrated advantage in this project, a real-time pipeline built primarily around the (cheaper) heuristic approach may be the more practical option.
 
### Better Visualization
Camera position and yaw are already extracted and used for evaluation calibration. The same data could drive a true minimap overlay (projecting player positions onto an actual in-game map image) rather than the camera-relative visualization used during development. This was scoped out of the current project's timeline due to time constraints but requires no new data collection, only reuse of the existing synchronization output.