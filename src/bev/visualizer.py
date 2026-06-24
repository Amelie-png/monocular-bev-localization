import cv2
import numpy as np

class BevVisualizer:
  def __init__(self, color=(0, 255, 0), scale=1):
    """
    Args:
      color: BGR color for players
    """
    self.color = color
    self.scale = scale
  
  def draw_players(self, estimations, height=800, width=800):
    """
    Draw camera player and other player position estimations.

    Args:
      estimations: List of estimation dicts from HeuristicBEVEstimator
      height: Height of output image
      width: Width of output image

    Return:
      Image with player positions
    """
    image = np.ones((height, width, 3), dtype=np.uint8) * 255
    # Draw camera (filled black triangle)
    camera_x = width // 2
    camera_y = height - 20
    pts = np.array([
      [camera_x, camera_y - 10],
      [camera_x - 10, camera_y + 10],
      [camera_x + 10, camera_y + 10]
    ])

    cv2.fillPoly(
      image,
      [pts],
      (0,0,0)
    )

    # Draw players (filled circle with visualizer color)
    for estimation in estimations:
      bev_x = estimation["bev_x"]
      bev_y = estimation["bev_y"]
      draw_x = int(camera_x + bev_x * self.scale)
      draw_y = int(camera_y - bev_y * self.scale)
      cv2.circle(image, (draw_x, draw_y), 4, self.color, -1)

    return image
  
  def create_bev_video(self, estimation_list, output_path, fps=30):
    """
    Create video with BEV estimation visualization.

    Args:
      estimation_list: List of estimation lists (one per frame)
      output_path: Where to save video
      fps: Frame rate
    """
    # Read first frame to get dimensions
    height, width = 800, 800 # set dimension here
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    
    for estimation in estimation_list:
      # Draw estimations
      image = self.draw_players(estimation, height, width)
      
      # Write to video
      out.write(image)
    
    out.release()
    print(f"BEV estimation video saved to: {output_path}")