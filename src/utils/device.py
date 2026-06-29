import torch

def detect_available_device():
  """
  Detect available compute device.
  
  Return:
    tuple: (device_str, device_name)
    e.g., ('cuda', 'NVIDIA GeForce RTX 3090')
    or    ('cpu', 'CPU')
  """
  if torch.cuda.is_available():
    gpu_name = torch.cuda.get_device_name(0)
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
    return 'cuda', f"GPU: {gpu_name} ({gpu_memory:.1f}GB)"
  elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    return "mps", "Apple Metal"
  else:
    return 'cpu', 'CPU'

def get_device_info():
  """
  Get detailed device information.
  """
  device, device_name = detect_available_device()
  
  info = {
    'device': device,
    'device_name': device_name,
    'torch_version': torch.__version__,
    'cuda_available': torch.cuda.is_available(),
    'mps_available': hasattr(torch.backends, "mps") and torch.backends.mps.is_available(),
  }
  
  if device == 'cuda':
    info['cuda_version'] = torch.version.cuda
    info['gpu_count'] = torch.cuda.device_count()

  if hasattr(torch.backends, "mps"):
    info["mps_built"] = torch.backends.mps.is_built()

  return info