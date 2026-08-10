from pathlib import Path
import subprocess
from tqdm import tqdm

def is_h264(path):
  """
  Check codec and re-encode only when format is not h264, already-fixed files are skipped.
  """
  result = subprocess.run(
    ["ffprobe", "-v", "error", "-select_streams", "v:0",
     "-show_entries", "stream=codec_name", "-of", "csv=p=0", str(path)],
    capture_output=True, text=True
  )
  return result.stdout.strip() == "h264"

def reencode_h264(path):
  path = Path(path)
  tmp_path = path.with_name(path.stem + "_h264tmp.mp4")
  cmd = ["ffmpeg", "-y", "-i", str(path), "-c:v", "libx264",
         "-pix_fmt", "yuv420p", "-crf", "18", str(tmp_path)]
  result = subprocess.run(cmd, capture_output=True)
  if result.returncode != 0:
    print(f"FAILED: {path}\n{result.stderr.decode()[-500:]}")
    tmp_path.unlink(missing_ok=True)
    return False
  path.unlink()
  tmp_path.rename(path)
  return True

def reencode_all(root="outputs/videos", skip_existing=True):
  root = Path(root)
  video_files = list(root.rglob("*.mp4"))
  print(f"Found {len(video_files)} video(s) under {root}")

  reencoded, skipped, failed = 0, 0, 0
  for path in tqdm(video_files, desc="Re-encoding", unit="video"):
    if skip_existing and is_h264(path):
      skipped += 1
      continue
    if reencode_h264(path):
      reencoded += 1
    else:
      failed += 1

  print(f"\nDone. Re-encoded: {reencoded}, already H.264 (skipped): {skipped}, failed: {failed}")

if __name__ == "__main__":
  import argparse
  parser = argparse.ArgumentParser()
  parser.add_argument("--root", type=str, default="outputs/videos", help="Directory to scan for .mp4 files")
  parser.add_argument("--force", action="store_true", help="Re-encode even files already in H.264")
  args = parser.parse_args()

  reencode_all(root=args.root, skip_existing=not args.force)