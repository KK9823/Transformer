import torch
import torch.nn as nn
from .attention import MultiHeadAttention

class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, block_size):
        super().__init__()

        # LayerNorm before attention
        self.ln1 = nn.LayerNorm(d_model)

        # Multi-head attention
        self.attn = MultiHeadAttention(d_model, n_heads, block_size)

        # LayerNorm before FFN
        self.ln2 = nn.LayerNorm(d_model)

        # Feed-forward network
        # has one hidden layer
        # data goes from d_model -> 4*d_model -> d_model dimensions at each layer
        self.ff = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),          # GELU activation function
            nn.Linear(4 * d_model, d_model)
        )

    def forward(self, x):
        # Normalize the input before each layer
        # This is one transformer block operation

        # Attention + residual
        x = x + self.attn(self.ln1(x))

        # FFN + residual
        x = x + self.ff(self.ln2(x))

        return x
