from trackers import ByteTrackTracker
from supervision import Detections
import numpy as np

class PlayerTracker:
  def __init__(self, config):
    self.tracker = ByteTrackTracker(
      track_activation_threshold=config.track_activation_threshold,
      lost_track_buffer=config.lost_track_buffer,
      minimum_consecutive_frames=config.minimum_consecutive_frames
    )

  def update(self, detections):
    """
    Update tracker with new detections.

    Args:
      detections: detections (List of dicts with bbox) in current frame given by YOLOv8

    Return:
      List of tracked objects with track_id
    """
    if len(detections) == 0:
      return []

    xyxy = []
    confidence = []

    xyxy = np.asarray(
      [d["bbox"] for d in detections],
      dtype=np.float32
    )

    confidence = np.asarray(
      [d["confidence"] for d in detections],
      dtype=np.float32
    )

    class_id = np.zeros(len(detections), dtype=np.int32)

    detections_sv = Detections(
      xyxy=xyxy,
      confidence=confidence,
      class_id=class_id
    )

    tracks = self.tracker.update(detections_sv)

    results = []

    for bbox, track_id, conf in zip(
      tracks.xyxy,
      tracks.tracker_id,
      tracks.confidence
    ):
      results.append({
        "track_id": int(track_id),
        "bbox": bbox,
        "confidence": float(conf)
      })

    return results
  
  def reset(self):
    self.tracker.reset()