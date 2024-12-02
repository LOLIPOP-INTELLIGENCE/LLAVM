from datasets import load_dataset
from pathlib import Path
from openai import OpenAI

client = OpenAI()

# Create audio directory if it doesn't exist
audio_dir = Path(__file__).parent / "audio"
audio_dir.mkdir(exist_ok=True)

def gen_audio(text, sample_id):
    speech_file_path = audio_dir / f"{sample_id}.mp3"
    response = client.audio.speech.create(
        model="tts-1",
        voice="alloy",
        input=text
    )
    response.stream_to_file(speech_file_path)

# Load and process dataset
dataset = load_dataset("lmms-lab/LLaVA-Video-178K", "30_60_s_youtube_v0_1")
random_samples = dataset['open_ended'].shuffle(seed=50).select(range(100))

# Generate audio files
for i, sample in enumerate(random_samples):
    print(f"\nProcessing sample {i+1}:")
    for conv in sample['conversations']:
        if conv['from'] == 'human':
            text = conv['value']
            # Use the sample's ID for the filename
            sample_id = sample['id']
            gen_audio(text=text, sample_id=sample_id)
            print(f"Generated audio for sample ID: {sample_id}")
            break