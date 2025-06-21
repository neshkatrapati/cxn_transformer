import torch
from torch.utils.data import Dataset, DataLoader
from collections import Counter

class ParallelTextDataset(Dataset):
    """Dataset for parallel text files."""
    def __init__(self, ):
        self.src_path = src_path
        self.tgt_path = tgt_path
        
        self.run(src_path, tgt_path, min_freq=1)
        
    def run(self, src_path, tgt_path, min_freq=1):
        with open(src_path, 'r', encoding='utf-8') as f:
            src_lines = [l.strip() for l in f if l.strip()]
        with open(tgt_path, 'r', encoding='utf-8') as f:
            tgt_lines = [l.strip() for l in f if l.strip()]
        assert len(src_lines) == len(tgt_lines), "Source and target files must have same number of lines"

        self.src_tokens = [l.split() for l in src_lines]
        self.tgt_tokens = [l.split() for l in tgt_lines]

        src_counter = Counter(tok for sent in self.src_tokens for tok in sent)
        tgt_counter = Counter(tok for sent in self.tgt_tokens for tok in sent)

        self.src_vocab = {'<pad>':0, '<unk>':1}
        for tok, freq in src_counter.items():
            if freq >= min_freq:
                self.src_vocab.setdefault(tok, len(self.src_vocab))

        self.tgt_vocab = {'<pad>':0, '<unk>':1, '<bos>':2, '<eos>':3}
        for tok, freq in tgt_counter.items():
            if freq >= min_freq:
                self.tgt_vocab.setdefault(tok, len(self.tgt_vocab))

        self.data = []
        for s_tokens, t_tokens in zip(self.src_tokens, self.tgt_tokens):
            src_ids = [self.src_vocab.get(tok, 1) for tok in s_tokens]
            tgt_ids = [2] + [self.tgt_vocab.get(tok, 1) for tok in t_tokens] + [3]
            self.data.append((src_ids, tgt_ids))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def collate_fn(batch):
    src_batch, tgt_batch = zip(*batch)
    src_len = max(len(s) for s in src_batch)
    tgt_len = max(len(t) for t in tgt_batch)
    padded_src = [s + [0]*(src_len - len(s)) for s in src_batch]
    padded_tgt = [t + [0]*(tgt_len - len(t)) for t in tgt_batch]
    return torch.tensor(padded_src), torch.tensor(padded_tgt)

def build_dataloader(src_path, tgt_path, batch_size=32, min_freq=1, shuffle=False):
    dataset = ParallelTextDataset(src_path, tgt_path, min_freq)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)
    return dataset, loader
