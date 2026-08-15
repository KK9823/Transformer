import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt

class Head(nn.Module):
    def __init__(self, d_model, d_head, block_size):
        super().__init__()
        self.d_head = d_head

        # Learned linear projections
        self.W_Q = nn.Linear(d_model, d_head, bias=False)
        self.W_K = nn.Linear(d_model, d_head, bias=False)
        self.W_V = nn.Linear(d_model, d_head, bias=False)

        # Causal mask (lower triangular)
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(block_size, block_size))
        )

    def forward(self, x):
        # x: (batch, block, d_model)

        Q = self.W_Q(x)  # (batch, block, d_head)
        K = self.W_K(x)  # (batch, block, d_head)
        V = self.W_V(x)  # (batch, block, d_head)

        # Compute raw scores: Q @ K^T
        # Q: (B, T, d_head)
        # K: (B, T, d_head)
        # scores: (B, T, T)
        scores = Q @ K.transpose(-2, -1)

        # Scale by sqrt(d_head)
        scores = scores / sqrt(self.d_head)

        # Apply causal mask
        # mask: (T, T) → broadcast to (B, T, T)
        scores = scores.masked_fill(self.mask == 0, float('-inf'))

        # Softmax → attention weights
        weights = F.softmax(scores, dim=-1)

        # Weighted sum of V
        out = weights @ V  # (B, T, d_head)

        return out


# MultiHead (Wraps Head)
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, block_size):
        super().__init__()
        d_head = d_model // n_heads

        self.heads = nn.ModuleList([
            Head(d_model, d_head, block_size)
            for _ in range(n_heads)
        ])

        # Final linear projection
        self.W_O = nn.Linear(d_model, d_model)

    def forward(self, x):
        # Run all heads
        head_outputs = [h(x) for h in self.heads]

        # Concatenate along the last dimension
        # Each head: (B, T, d_head)
        # After concat: (B, T, n_heads * d_head = d_model)
        out = torch.cat(head_outputs, dim=-1)

        # Final projection
        out = self.W_O(out)

        return out

