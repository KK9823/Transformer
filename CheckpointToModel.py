# This file can be used to turn a checkpoint_n.pt into model.pt for testing
# The resulting model.pt can be tested with main.py

import torch
from models.transformer import Transformer
from utils.CharTokenizer import CharTokenizer
import config

model = Transformer(
    config.vocab_size,
    config.d_model,
    config.n_heads,
    config.n_layers,
    config.block_size
)

name = input("Enter checkpoint name to load (exclude .pt): ")
checkpoint = torch.load(f"{name}.pt", map_location=config.device)

model.load_state_dict(checkpoint['model'])
torch.save(model.state_dict(), "model.pt")

print("Successfully written to model.pt")