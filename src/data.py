import torch
from torch.utils.data import Dataset, DataLoader
from collections import Counter

class TextDataset(Dataset):
    def __init__(self, path, seq_len=32, min_freq=1):
        with open(path, 'r', encoding='utf-8') as f:
            text = f.read()
        tokens = text.split()
        counter = Counter(tokens)
        self.vocab = {
            token: i + 2
            for i, (token, freq) in enumerate(counter.items())
            if freq >= min_freq
        }
        self.vocab['<pad>'] = 0
        self.vocab['<unk>'] = 1
        self.inv_vocab = {i: t for t, i in self.vocab.items()}
        ids = [self.vocab.get(tok, 1) for tok in tokens]
        self.seq_len = seq_len
        self.data = []
        for i in range(0, len(ids) - seq_len):
            self.data.append((ids[i:i+seq_len], ids[i+1:i+seq_len+1]))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        src, tgt = self.data[idx]
        return torch.tensor(src), torch.tensor(tgt)


def build_dataloader(path, seq_len=32, batch_size=32, min_freq=1):
    dataset = TextDataset(path, seq_len, min_freq)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataset, loader
