import torch
from whisperspeech.t2s_up_wds_mlang_enclm import TSARTransformer

# First load the model
model = TSARTransformer.load_model("collabora/whisperspeech:t2s-v1.9-medium-7lang.model")
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.ensure_tokenizer()

def text_to_tokens_to_embeddings(text, cps=15, language="en"):
    """
    3-step process:
    1. Convert text to tokens
    2. Get semantic tokens
    3. Convert semantic tokens to embeddings
    """
    # Step 1 & 2: Text to semantic tokens using the model
    ttoks, cpss, langs = model.prep(text, cps=cps, lang=language)
    with torch.no_grad():
        # Generate semantic tokens
        semantic_tokens = model.generate(text, cps=cps, lang=language)
        print(f"Semantic tokens shape: {semantic_tokens.shape}")
        
        # Step 3: Convert semantic tokens to embeddings using T2SEmbedding
        # We need xenc for dtype matching, so let's get it
        xenc, _, cps_emb = model.run_encoder(ttoks, langs, cpss)
        
        # Now use the model's embeddings to convert tokens to embeddings
        semantic_embeddings, _ = model.embeddings(semantic_tokens, xenc, cps_emb)
        print(f"Semantic embeddings shape: {semantic_embeddings.shape}")
        
        return semantic_tokens, semantic_embeddings

# Example usage
text = "Hello world"
tokens, embeddings = text_to_tokens_to_embeddings(text)
import pdb; pdb.set_trace()