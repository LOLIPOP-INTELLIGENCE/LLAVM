import torch
from whisperspeech.t2s_up_wds_mlang_enclm import TSARTransformer
from whisperspeech.vq_stoks import RQBottleneckTransformer, make_model
from huggingface_hub import hf_hub_download
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def fix_state_dict(state_dict):
    new_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith('out_blocks'):
            new_key = '_' + key  # Add underscore
        elif key.startswith('rq.project'):
            # Convert rq.project to rq.layers.0.project
            new_key = key.replace('rq.project', 'rq.layers.0.project')
        else:
            new_key = key
        new_state_dict[new_key] = value
    return new_state_dict

def audio_to_semantic_embeddings(audio_path, vq_model=None, t2s_model=None):
    if vq_model is None:
        # Load the VQ model for audio->semantic tokens
        vq_model = RQBottleneckTransformer() # Create empty model first
        vq_model.load_model()
        vq_model.eval()
        vq_model.ensure_whisper(device)

    if t2s_model is None:
        # Load the T2S model for semantic tokens->embeddings
        t2s_model = TSARTransformer.load_model("collabora/whisperspeech:t2s-v1.9-medium-7lang.model")
        t2s_model.eval()
        t2s_model.ensure_tokenizer()

    # Step 1: Audio to semantic tokens
    semantic_tokens = vq_model.encode_audio(audio_path)
    print(f"Semantic tokens shape: {semantic_tokens.shape}")

    # Step 2: Semantic tokens to embeddings 
    # We need some encoder output for dtype matching, so let's create dummy inputs
    dtype = next(t2s_model.parameters()).dtype
    semantic_tokens = semantic_tokens.to(dtype)
    xenc = torch.zeros((1, 1, t2s_model.width), dtype=dtype, device=device)
    cps_emb = None

    # Now convert tokens to embeddings
    semantic_embeddings, _ = t2s_model.embeddings(semantic_tokens, xenc, cps_emb)
    print(f"Semantic embeddings shape: {semantic_embeddings.shape}")

    return semantic_tokens, semantic_embeddings

# Example usage
audio_path = "input.wav"
tokens, embeddings = audio_to_semantic_embeddings(audio_path)
import pdb; pdb.set_trace()