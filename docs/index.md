# Monocular BEV Localization

---

## Abstract

Text goes here.

---

## Motivation

## Project objectives / Key features

## Pipeline Overview
```mermaid
flowchart LR
A[Demo] --> B[Frame Extraction]
B --> C[YOLO Detection]
C --> D[DeepSORT]
D --> E[BEV Localization]
E --> F[Visualization]
```
<figure markdown>
  ![Pipeline overview](assets/pipeline.png){ width="800" }
  <figcaption>End-to-end pipeline: detection → tracking → BEV projection</figcaption>
</figure>

## Project Structure

## Sample Output

## Acknowledgements / Credits