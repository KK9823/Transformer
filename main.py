import torch
from models.transformer import Transformer
import config

transformer = Transformer(
    config.vocab_size,
    config.d_model,
    config.n_heads,
    config.n_layers,
    config.block_size
)


idx = torch.randint(0, config.vocab_size, (2, config.block_size))
logits = transformer(idx)
print(logits.shape)  # should be (2, block_size, vocab_size)
