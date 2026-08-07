import torch

# Takes the data source and then generates a random batch from the source
def get_batch(data, block_size, batch_size):
    n = len(data)

    # Random starting positions
    ix = torch.randint(0, n - block_size - 1, (batch_size,))

    # Build x and y batches
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])

    return x, y
