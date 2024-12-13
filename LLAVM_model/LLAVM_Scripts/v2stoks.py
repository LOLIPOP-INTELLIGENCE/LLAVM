from whisperspeech.t2s_up_wds_mlang_enclm import TSARTransformer

class TTSemanticToken:
    def __init__(self, device: str) -> None:
        self.t2s_model = TSARTransformer.load_model("collabora/whisperspeech:t2s-v1.9-medium-7lang.model", device = device)
        self.t2s_model.optimize(torch_compile=True)
    def convert_text_to_semantic(self,text: str):
        """
            Convert texts to semantic tokens
        Args:
            text (str): The text to convert to audio
        Returns:
            torch.Tensor: The generated audio
        """
        return self.t2s_model.generate(text, lang="en",cps=15, T =0.0, step=None)

