import torch
from whisperspeech.t2s_up_wds_mlang_enclm import TSARTransformer
from whisperspeech.vq_stoks import RQBottleneckTransformer, make_model
from huggingface_hub import hf_hub_download
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def audio_to_semantic_embeddings(audio_path, vq_model=None, t2s_model=None):
    if vq_model is None:
        # Load the VQ model for audio->semantic tokens

        import os
        # if not os.path.exists("whisper-vq-stoks-v3-7lang-fixed.model"):
        hf_hub_download(
        repo_id="jan-hq/WhisperVQ",
        filename="whisper-vq-stoks-v3-7lang-fixed.model",
        local_dir=".",
        )
        vq_model = RQBottleneckTransformer.load_model(
                "whisper-vq-stoks-v3-7lang-fixed.model"
            ).to(device)
        vq_model.ensure_whisper(device)

    if t2s_model is None:
        # Load the T2S model for semantic tokens->embeddings
        t2s_model = TSARTransformer.load_model("collabora/whisperspeech:t2s-v1.9-medium-7lang.model").to(device)
        t2s_model.eval()
        t2s_model.ensure_tokenizer()

    # Step 1: Audio to semantic tokens
    semantic_tokens = vq_model.encode_audio(audio_path)
    print(f"Semantic tokens shape: {semantic_tokens.shape}")

    # Step 2: Semantic tokens to embeddings 
    # We need some encoder output for dtype matching, so let's create dummy inputs
    ttoks, cpss, langs = t2s_model.prep("dummy text")  # Create dummy input
    xenc, _, cps_emb = t2s_model.run_encoder(ttoks, langs, cpss)

    # Now convert tokens to embeddings
    semantic_embeddings, _ = t2s_model.embeddings(semantic_tokens, xenc, cps_emb)
    print(f"Semantic embeddings shape: {semantic_embeddings.shape}")

    return semantic_tokens, semantic_embeddings

# Example usage
audio_path = "input.wav"
tokens, embeddings = audio_to_semantic_embeddings(audio_path)