from whisperspeech.vqstoks import RQBottleneckTransformer
import torch
import torch.nn as nn
import torch.nn.functional as F

class AudioProjector(nn.Module):
    def __init__(self, num_tokens, emb_dim, num_heads, num_layers):
        super().__init__()
        self.tokenizer = RQBottleneckTransformer(num_tokens, emb_dim, num_heads, num_layers)
        self.proj = MLP(emb_dim)
        self.num_tokens = num_tokens
        self.emb_dim = emb_dim
    def forward(self, x):
        x = self.tokenizer(x)
        x = self.proj(x)
        return x
    def save(self, path):
        torch.save(self.state_dict(), path)
    def load(self, path):
        self.load_state_dict(torch.load(path))


class MLP(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.proj = nn.Linear(emb_dim, 4*emb_dim)
        self.gelu = nn.GELU()
        self.proj2 = nn.Linear(emb_dim *4, emb_dim)
        self.gelu2 = nn.GELU()

    def forward(self, x):
        x = self.proj(x)
        x = self.gelu(x)
        x = self.proj2(x)
        x = self.gelu2(x)
        return x

