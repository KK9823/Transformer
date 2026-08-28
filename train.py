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
data = torch.load("data/data.pt")

option = input("Load an existing model?: ")

# Create the model
model = Transformer(
    config.vocab_size,
    config.d_model,
    config.n_heads,
    config.n_layers,
    config.block_size
)
model.to(config.device)

if option.lower() == "no" or option.lower() == "n":
    start_step = 0

else:
    name = input("Enter model name to load (exclude .pt): ")
    checkpoint = torch.load(f"{name}.pt", map_location=config.device)

    model.load_state_dict(checkpoint['model'])

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    optimizer.load_state_dict(checkpoint['optimizer'])

    start_step = checkpoint['step']

# Optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)

# Separate the data into training and evaluation
n = len(data)
positions = torch.arange(0, n - config.block_size - 1)
positions = positions[torch.randperm(len(positions))]

# Basically shuffle all the possible start positions and group most into training and rest into evaluation
split = int(0.9 * len(positions))
train_positions = positions[:split]
val_positions   = positions[split:]

try:
    for step in range(start_step, config.max_steps):

        x,y = get_batch(data, train_positions, config.block_size, config.batch_size)
        x = x.to(config.device)
        y = y.to(config.device)

        logits = model(x)
        loss = compute_loss(logits, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 50 == 0:
            print(f"step {step} | train loss {loss.item():.4f}")

        if step % config.eval_interval == 0:
            with torch.no_grad():
                x, y = get_batch(data, val_positions, config.block_size, config.batch_size)
                x = x.to(config.device)
                y = y.to(config.device)
                logits_val = model(x)
                val_loss = compute_loss(logits_val, y)
                print(f"step {step} | val loss {val_loss.item():.4f}")

except KeyboardInterrupt:
    print("Interrupted. Saving the model...")
    torch.save({
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'step': step,
    }, f"checkpoint_{step}.pt")
    print(f"Model Saved as checkpoint_{step}.pt")

torch.save(model.state_dict(), "model.pt")