import os
import torchaudio
from torchaudio.transforms import Resample

def resample_audio_to_16k(input_dir, output_dir):
    """
    Resamples all audio files in the input directory to 16 kHz and saves them in the output directory.

    Args:
        input_dir (str): Path to the input directory containing audio files.
        output_dir (str): Path to the output directory to save resampled audio files.
    """
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    for file_name in os.listdir(input_dir):
        if file_name.endswith(".mp3"):
            input_path = os.path.join(input_dir, file_name)
            output_path = os.path.join(output_dir, file_name)

            # Load the audio file
            waveform, sample_rate = torchaudio.load(input_path)

            # Check if resampling is necessary
            if sample_rate != 16000:
                resampler = Resample(orig_freq=sample_rate, new_freq=16000)
                waveform = resampler(waveform)

            # Save the resampled audio
            torchaudio.save(output_path, waveform, sample_rate=16000)
            print(f"Resampled: {file_name} -> {output_path}")

# Define input and output directories
input_directory = "audio"
output_directory = "resampled_audio"

# Resample audio files
resample_audio_to_16k(input_directory, output_directory)

