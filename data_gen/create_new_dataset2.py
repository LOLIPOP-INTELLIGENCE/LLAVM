from datasets import load_dataset
import os
from pathlib import Path
import pdb

# Load dataset
dataset = load_dataset("lmms-lab/LLaVA-Video-178K", "0_30_s_academic_v0_1", split="caption")

# Get available audio files - now recursively searching subdirectories
audio_dir = Path("academic_source")
available_audio_files = {
    f.stem: f for f in audio_dir.rglob("*.mp4")
}

print(len(available_audio_files))


# Filter dataset to only rows with audio
def has_audio(row):
    # Check if the ID exists in our available files
    return row['id'] in available_audio_files

final_dataset = dataset.filter(has_audio)
print(len(final_dataset))
# Save locally
save_path = "processed_dataset"
os.makedirs(save_path, exist_ok=True)
final_dataset.save_to_disk(save_path)
