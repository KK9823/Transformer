from utils.CharTokenizer import CharTokenizer
import torch

with open("data/combined.txt", "r", encoding="utf-8") as f:
    text = f.read()

# Tokenize the text
tokenizer = CharTokenizer(text)
print(tokenizer.vocab_size())                       # Get the vocab size for config
ids = tokenizer.encode(text)
data = torch.tensor(ids, dtype=torch.long)

# Separate into training and testing data (90/10 split)
n = len(data)
train_data = data[:int(0.9*n)]
val_data = data[int(0.9*n):]

# Save the data
torch.save(train_data, "data/train.pt")
torch.save(val_data, "data/val.pt")