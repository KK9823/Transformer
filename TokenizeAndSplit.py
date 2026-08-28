from utils.CharTokenizer import CharTokenizer
import torch
import config

with open("data/combined.txt", "r", encoding="utf-8") as f:
    text = f.read()

# Tokenize the text
tokenizer = CharTokenizer(text)
print(tokenizer.vocab_size())                       # Get the vocab size for config
ids = tokenizer.encode(text)
data = torch.tensor(ids, dtype=torch.long)

torch.save(data, "data/data.pt")