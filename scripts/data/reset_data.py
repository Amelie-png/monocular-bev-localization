from pathlib import Path
import shutil

folders_to_clear = [
  "data/processed",
  "outputs"
]

for folder in folders_to_clear:
  folder_path = Path(folder)

  if folder_path.exists():
    print(f"Deleting {folder}")
    shutil.rmtree(folder_path)

    folder_path.mkdir(parents=True, exist_ok=True)

print("Reset complete")