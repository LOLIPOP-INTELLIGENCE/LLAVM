import torch.nn as nn
import torch.functional as F


class SemanticProjection(nn.Module):
    def __init__(
        self,
        token_dim,
        embedding_dim,
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




        

