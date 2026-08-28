import torch

# Takes the data source and then generates a random batch from the source
# Returns: x - input batch (batch_size, block_size)
#          y - expected output batch (batch_size, block_size)
def get_batch(data, positions, block_size, batch_size):
    ix = positions[torch.randint(0, len(positions), (batch_size,))]
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x, y
