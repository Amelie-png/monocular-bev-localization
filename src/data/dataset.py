import pandas as pd
import cv2
from torch.utils.data import Dataset

class CS2DetectionDataset(Dataset):
  """
  PyTorch Dataset for CS2 frames with ground truth player positions.
  """
  def __init__(self, sync_file, position_file, transform=None):
    """
    Args:
      sync_file: Path to sync parquet (frame_id -> tick mapping)
      position_file: Path to position parquet (tick -> player positions)
      transform: Optional image transforms
    """
    self.sync_df = pd.read_parquet(sync_file)
    self.positions_df = pd.read_parquet(position_file)
    self.transform = transform
      
  def __len__(self):
    return len(self.sync_df)
  
  def __getitem__(self, idx):
    # Get frame info
    frame_info = self.sync_df.iloc[idx]
    frame_path = frame_info['frame_path']
    matched_tick = frame_info['matched_tick']
    
    # Load image
    image = cv2.imread(frame_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Get player positions at this tick
    positions = self.positions_df[
      self.positions_df['tick'] == matched_tick
    ][['X', 'Y', 'Z', 'player_name']].to_dict('records')
    
    sample = {
      'image': image,
      'positions': positions,
      'tick': matched_tick,
      'frame_id': frame_info['frame_id']
    }
    
    if self.transform:
      sample['image'] = self.transform(sample['image'])
    
    return sample