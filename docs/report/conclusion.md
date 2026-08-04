# Conclusion
 
This project set out to answer the following questions:

- How accurately can player positions be approximated in a bird’s-eye-view representation using only monocular RGB video input?
- How do simple geometric heuristics compare to learned monocular depth estimation models for BEV reconstruction using first-person footage?
- To what extent does temporal tracking improve the stability and accuracy of spatial estimates?
 
A complete pipeline was built and evaluated end-to-end:

```pipeline
A[Detection]
---
tooltip: YOLO26m
color: blue
---

B[Tracking]
---
tooltip: ByteTrack
color: teal
---

C[Depth estimation]
---
tooltip: Heuristic geometric and MiDaS-based
color: blue
---

D[BEV projection]
---
tooltip: BEV projection
color: purple
---

E[Trajectory construction]
---
tooltip: Trajectory construction
color: wine
---

A --> B
B --> C
C --> D
D --> E
```

with an evaluation framework aligning predictions to ground truth extracted directly from game demo files, using three complementary metrics (Euclidean positional error, relative spatial accuracy, trajectory consistency).
 
The central empirical finding was that **the simple geometric heuristic consistently and substantially outperformed the learned MiDaS-based approach** across all three metrics and all evaluated videos. Results suggested that it is not due to a fixable calibration issue (parameter tuning targeting MiDaS's normalization produced negligible improvement), but most likely due to frame-to-frame depth-estimation instability. This instability likely caused by applying a general-purpose monocular depth model independently per frame, without temporal consistency constraints.
 
*[Track-vs-no-track finding to be added to this summary once results are available.]*
 
## Future Work

### Larger Datasets
Only 9 of 64 available rounds were evaluated due to hardware constraints. Expanding to the full round set, and to additional matches/maps or other video sources beyond this project's dataset, would both strengthen confidence in the heuristic-vs-MiDaS finding and test its generalization.
 
### Better Detectors
Fine-tuning YOLO26 on domain-specific labeled data (rather than a general person detector) could improve recall under occlusion and reduce sensitivity to non-standard player poses (crouching, prone) that a general-purpose detector may not represent well in its training distribution.
 
### Better Tracking
Resolving the ambiguous/ID-switched tracks with better tracking models and improving manual track ID label precision and accuracy to reliably produce usable ground truth. Since appearance-based re-identification is largely unavailable in this project, this likely requires stronger motion modeling or exploiting game-specific cues (e.g. team-side constraints) rather than visual appearance.
 
### Learned BEV Models
Improving depth extraction using general-purpose monocular depth model by using a newer and more <abbr title="State-Of-The-Art">SOTA</abbr> model. Additionally, incorporating temporal consistency directly into training, rather than treating each frame independently as a next step.
 
### Real-Time Inference
The current pipeline is fully offline/batch. MiDaS depth extraction in particular is the most computationally expensive stage per frame. Given the heuristic method's demonstrated advantage in this project, a real-time pipeline built primarily around the (cheaper) heuristic approach may be the more practical option.
 
### Better Visualization
Camera position and yaw are already extracted and used for evaluation calibration. The same data could drive a true minimap overlay (projecting player positions onto an actual in-game map image) rather than the camera-relative visualization used during development. This was scoped out of the current project's timeline due to time constraints but requires no new data collection, only reuse of the existing synchronization output.