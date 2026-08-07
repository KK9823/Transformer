import torch
import torch.nn as nn

class Embeddings(nn.Module):
    def __init__(self, vocab_size, d_model, block_size):
        super().__init__()

        # Convert the character ids (ints) into multidimensional (d_model dimensions) vectors
        self.token_embedding = nn.Embedding(vocab_size, d_model)

        # Give characters meanings based on their position relative to other characters
        self.position_embedding = nn.Embedding(block_size, d_model)


    # Basically the "vectorization" layer
    # Takes in a tensor of batch_size * block_size
    # Returns a tensor of batch_size * block_size * d_model
    # essentially converting the characters from ids into multidimensional vectors
    def forward(self, idx):
        # idx shape: (batch_size, block_size)
        batch_size, seq_len = idx.shape

        token_emb = self.token_embedding(idx)  # (batch_size, block_size, d_model)

        # positions: [0, 1, 2, ..., seq_len-1]
        positions = torch.arange(seq_len, device=idx.device)
        pos_emb = self.position_embedding(positions)  # (block_size, d_model)

        # broadcast pos_emb across batch dimension
        pos_emb = pos_emb.unsqueeze(0)  # (1, block_size, d_model)

        return token_emb + pos_emb
