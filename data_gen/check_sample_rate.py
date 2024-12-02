import torchaudio
import os

def load_mp3_and_check_sample_rate(file_path):
    """
    Load an MP3 file as a PyTorch tensor and check its sample rate.
    
    Args:
        file_path (str): Path to the MP3 file.

    Returns:
        waveform (torch.Tensor): The audio waveform tensor.
        sample_rate (int): The sample rate of the audio.
    """
    waveform, sample_rate = torchaudio.load(file_path)
    print(f"Loaded {file_path}")
    print(f"Sample Rate: {sample_rate}")
    print(f"Waveform Shape: {waveform.shape}")
    assert sample_rate==16000
def process_directory(directory):
    """
    Load all MP3 files in a directory and check their sample rates.
    
    Args:
        directory (str): Path to the directory containing MP3 files.
    """
    for file_name in os.listdir(directory):
        if file_name.endswith(".mp3"):
            file_path = os.path.join(directory, file_name)
            load_mp3_and_check_sample_rate(file_path)

# Example Usage:
# Replace 'your_directory_path' with the path to the directory containing your MP3 files.
directory_path = "resampled_audio"
process_directory(directory_path)

