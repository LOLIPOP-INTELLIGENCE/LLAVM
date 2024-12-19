from transformers import AutoTokenizer
CONTROLLER_HEART_BEAT_EXPIRATION = 30
WORKER_HEART_BEAT_INTERVAL = 15

LOGDIR = "."

# Model Constants
IGNORE_INDEX = -100
IMAGE_TOKEN_INDEX = -200
AUDIO_TOKEN_INDEX = -300
DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_IMAGE_PATCH_TOKEN = "<im_patch>"
DEFAULT_IM_START_TOKEN = "<im_start>"
DEFAULT_IM_END_TOKEN = "<im_end>"
DEFAULT_AUDIO_TOKEN = "<sound>"
DEFAULT_AUDIO_PATCH_TOKEN = "<sound_patch>"
DEFAULT_AUDIO_START_TOKEN = "<|sound_start|>"
DEFAULT_AUDIO_END_TOKEN= "<|sound_end|>"

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-7B", trust_remote_code=True, padding_side='right')
IM_START_EMBED, IM_END_EMBED = tokenizer.additional_special_tokens_ids 