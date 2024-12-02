from datasets import load_dataset
import os
from pathlib import Path

# Load dataset
dataset = load_dataset("lmms-lab/LLaVA-Video-178K", split="train")

# Get available audio files
audio_dir = Path("resampled_audio")
available_audio_files = {f.stem: f for f in audio_dir.glob("*.mp3")}

# Filter dataset to only rows with audio
def has_audio(row):
    return row['id'] in available_audio_files

filtered_dataset = dataset.filter(has_audio)

# Add audio path
def add_audio_path(row):
    row['audio_path'] = str(available_audio_files[row['id']])
    return row

final_dataset = filtered_dataset.map(add_audio_path)
# Save locally
save_path = "processed_dataset"
os.makedirs(save_path, exist_ok=True)
final_dataset.save_to_disk(save_path)
