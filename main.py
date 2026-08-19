import torch

from models.transformer import Transformer
from utils.CharTokenizer import CharTokenizer
import config

# This reading thing every single run might be inefficient, maybe find a way to store the tokenizer later
with open("data/combined.txt", "r", encoding="utf-8") as f:
    text = f.read()

# Tokenize the text
tokenizer = CharTokenizer(text)


model = Transformer(
    config.vocab_size,
    config.d_model,
    config.n_heads,
    config.n_layers,
    config.block_size
)

model.load_state_dict(torch.load("model.pt"))
model.eval()
model.to("cuda")

prompt = input("Enter prompt: ")
# idx : (1, prompt length)
idx = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long).to("cuda")

for _ in range(200):  # generate 200 characters
    # If sequence is longer than block_size, crop it
    idx_cond = idx[:, -config.block_size:]

    # Forward pass
    logits = model(idx_cond)

    # Get logits for the last position
    next_logits = logits[:, -1, :]  # (1, vocab_size)

    # Convert to probabilities
    probs = torch.softmax(next_logits, dim=-1)

    # Sample from the distribution
    next_token = torch.multinomial(probs, num_samples=1)

    # Append to sequence
    idx = torch.cat([idx, next_token], dim=1)

output_text = tokenizer.decode(idx[0].tolist())
print(output_text)
