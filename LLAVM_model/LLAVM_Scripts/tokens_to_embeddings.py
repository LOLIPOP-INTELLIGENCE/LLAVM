import torch
from whisperspeech.t2s_up_wds_mlang_enclm import TSARTransformer

class TokensToEmbeddings:
    def __init__(self, model_path="collabora/whisperspeech:t2s-v1.9-medium-7lang.model"):
        # Initialize model
        self.model = TSARTransformer.load_model(model_path)
        self.model.eval()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def convert_to_embeddings(self, semantic_tokens, ttoks, langs, cpss):
        """
        Convert semantic tokens to embeddings
        
        Args:
            semantic_tokens (torch.Tensor): Semantic tokens from text_to_tokens
            ttoks (torch.Tensor): Text tokens from the preparation step
            langs (torch.Tensor): Language tokens from the preparation step
            cpss (torch.Tensor): Characters per second tokens from the preparation step
            
        Returns:
            torch.Tensor: Semantic embeddings
        """
        with torch.no_grad():
            # Get encoder output for dtype matching
            xenc, _, cps_emb = self.model.run_encoder(ttoks, langs, cpss)
            
            # Convert tokens to embeddings
            semantic_embeddings, _ = self.model.embeddings(semantic_tokens, xenc, cps_emb)
            print(f"Semantic embeddings shape: {semantic_embeddings.shape}")
            
            return semantic_embeddings

# Example usage showing how to use both classes together
if __name__ == "__main__":
    from text_to_tokens import TextToTokens
    
    # Initialize both converters
    token_converter = TextToTokens()
    embedding_converter = TokensToEmbeddings()
    
    # Convert text to tokens
    text = "Hello world"
    tokens, ttoks, cpss, langs = token_converter.generate_semantic_tokens(text)
    
    # Convert tokens to embeddings
    embeddings = embedding_converter.convert_to_embeddings(tokens, ttoks, langs, cpss)
    print(f"Generated embeddings for '{text}'")