import torch
from models.transformer_block import TransformerBlock
import config

block = TransformerBlock(
    d_model=config.d_model,
    n_heads=config.n_heads,
    block_size=config.block_size
)

x = torch.randn(2, config.block_size, config.d_model)  # (batch, block, d_model)
out = block(x)

print(out.shape)
