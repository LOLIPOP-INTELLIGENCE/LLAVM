import torch
from whisperspeech.t2s_up_wds_mlang_enclm import TSARTransformer

class TextToTokens:
    def __init__(self, model_path="collabora/whisperspeech:t2s-v1.9-medium-7lang.model"):
        # Initialize model
        self.model = TSARTransformer.load_model(model_path)
        self.model.eval()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.ensure_tokenizer()

    def generate_semantic_tokens(self, text, cps=15, language="en"):
        """
        Convert text to semantic tokens
        
        Args:
            text (str): Input text to convert
            cps (int): Characters per second (speech rate)
            language (str): Language code (e.g., "en" for English)
            
        Returns:
            tuple: (semantic_tokens, ttoks, cpss, langs) - The semantic tokens and additional context
        """
        # Prepare input tensors
        ttoks, cpss, langs = self.model.prep(text, cps=cps, lang=language)
        
        with torch.no_grad():
            # Generate semantic tokens
            semantic_tokens = self.model.generate(text, cps=cps, lang=language)
            print(f"Semantic tokens shape: {semantic_tokens.shape}")
            
            return semantic_tokens, ttoks, cpss, langs

# Example usage
if __name__ == "__main__":
    converter = TextToTokens()
    text = "Hello world"
    tokens, *_ = converter.generate_semantic_tokens(text)
    print(f"Generated tokens for '{text}'")
    import pdb; pdb.set_trace()