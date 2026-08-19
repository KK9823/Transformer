import torch
import torch.nn.functional

from config import vocab_size
from models.transformer import Transformer
from utils.CharTokenizer import CharTokenizer
from utils.GetBatch import get_batch
import config

def compute_loss(logits, y):
    logits = logits.reshape(-1, vocab_size)
    targets = y.reshape(-1)
    return torch.nn.functional.cross_entropy(logits, targets)

# Load the data
train_data = torch.load("data/train.pt")
val_data = torch.load("data/val.pt")

# Create the model
model = Transformer(
    config.vocab_size,
    config.d_model,
    config.n_heads,
    config.n_layers,
    config.block_size
)
model.to(config.device)

# Optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)

# Training
max_steps = 5000
eval_interval = 200

for step in range(max_steps):

    x,y = get_batch(train_data, config.block_size, config.batch_size)
    x = x.to(config.device)
    y = y.to(config.device)

    logits = model(x)
    loss = compute_loss(logits, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if step % 50 == 0:
        print(f"step {step} | train loss {loss.item():.4f}")

    if step % eval_interval == 0:
        with torch.no_grad():
            x, y = get_batch(val_data, config.block_size, config.batch_size)
            x = x.to(config.device)
            y = y.to(config.device)
            logits_val = model(x)
            val_loss = compute_loss(logits_val, y)
            print(f"step {step} | val loss {val_loss.item():.4f}")

# Save
torch.save(model.state_dict(), "model.pt")