import torch
import torch.nn as nn

from .embeddings import Embeddings
from .transformer_block import TransformerBlock


class Transformer(nn.Module):
    def __init__(self, vocab_size, d_model, n_heads, n_layers, block_size):
        super().__init__()

        # Combined token + positional embeddings
        self.emb = Embeddings(vocab_size, d_model, block_size)

        # Stack of transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, block_size)
            for _ in range(n_layers)
        ])

        # Final layer norm
        self.ln_f = nn.LayerNorm(d_model)

        # Output projection to vocabulary
        self.lm_head = nn.Linear(d_model, vocab_size)

        # Store config (in case its needed later?)
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.block_size = block_size

    def forward(self, idx):
        """
        idx: (batch, block) — integer token IDs
        returns: logits of shape (batch, block, vocab_size)
        """

        # 1. Embeddings (token + positional)
        # x: (batch, block, d_model)
        x = self.emb(idx)

        # 2. Transformer blocks
        for block in self.blocks:
            x = block(x)

        # 3. Final layer norm
        x = self.ln_f(x)

        # 4. Output logits
        logits = self.lm_head(x)  # (batch, block, vocab_size)

        return logits
