class CharTokenizer:
    def __init__(self, text):
        chars = sorted(list(set(text)))

        # Map the individual characters into integers
        # (Also make another dict that goes the other way around)
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for ch, i in self.stoi.items()}

    def encode(self, s):
        return [self.stoi[ch] for ch in s]

    def decode(self, ids):
        return "".join(self.itos[i] for i in ids)

    def vocab_size(self):
        return len(self.stoi)
