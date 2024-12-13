import torch.nn as nn
import torch.functional as F


# batch_size * token_length * 1024
# 1024 * 1024

class SemanticProjection(nn.Module):
    def __init__(
        self,
        token_dim=1024,
        embedding_dim=3584,
        init_scale: float = 0.02
                 ):
        super().__init__()
        self.projection = nn.Linear(token_dim, embedding_dim)

    def init_weights(self, init_scale: float): #Unused helper func in case glorot or kaiming is more beneficial
        nn.init.normal_(self.projection.weight, mean = 0.0, std = init_scale)
        if self.projection.bias is not None:
            nn.init.zeros_(self.projection.bias)


    def forward(self, x):
        return self.projection(x)




        

