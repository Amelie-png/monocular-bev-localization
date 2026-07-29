from scripts.data.extract_frames import run_frame_extraction
from scripts.detection.run_detection import run_detection
from scripts.tracking.run_tracking import run_tracking
from scripts.bev.run_depth import run_depth
from scripts.bev.run_bev import run_bev
from scripts.utils.build_trajectory import build_trajectory
from scripts.utils.config_loader import load_pipeline_config
from src.utils import load_split_file

def run_pipeline(config, videos, force=False):
  print(f"\n{'='*60}")
  print(f"Running complete pipeline on {len(videos)} video(s)")
  print(f"{'='*60}")

  run_frame_extraction(fps=None, video_names=videos, force=force) # TODO add fps config
  run_detection(config=config.detection, video_names=videos, force=force)
  run_tracking(config=config.tracking, video_names=videos, force=force)
  run_depth(video_names=videos, force=force)
  run_bev(config=config.heuristic_bev, video_names=videos, midas=False, force=force)
  run_bev(config=config.midas_bev, video_names=videos, midas=True, force=force)

  build_trajectory(videos, variant="heuristic", force=force)
  build_trajectory(videos, variant="midas", force=force)

  # TODO trajectory + visualization
  """
  for v in videos:
    for variant in ("heuristic", "midas"):
      build_trajectories(v, bev_variant=variant)
      draw_trajectories(
        pitch_template_path="assets/pitch_template.png",
        trajectories_path=f"outputs/trajectories/{v}_{variant}.json",
        out_path=f"outputs/trajectories/{v}_{variant}.png",
      )

    combine_videos_grid(
      [f"outputs/viz/{v}_tracked.mp4",
        f"outputs/viz/{v}_bev_heuristic.mp4",
        f"outputs/viz/{v}_bev_midas.mp4"],
      f"outputs/viz/{v}_combined.mp4",
      layout="hstack",
    )
  """

  print("\nPipeline complete.")

if __name__ == "__main__":
  config = load_pipeline_config("configs")
  train_videos = load_split_file("data/splits/train.txt")
  run_pipeline(config, train_videos)
