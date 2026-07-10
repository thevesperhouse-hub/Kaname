"""Datasets.

BinTokenDataset: a flat token stream memmapped from disk (uint32 because vocab=100k
exceeds uint16). Yields (input_ids, targets) windows of length seq_len with next-token
shift. RandomTokenDataset: synthetic tokens for smoke tests / plumbing checks.
"""

import numpy as np
import torch
from torch.utils.data import Dataset


class RandomTokenDataset(Dataset):
    def __init__(self, vocab_size: int, seq_len: int, n_samples: int = 1024, seed: int = 0):
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.n_samples = n_samples
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        toks = self.rng.integers(0, self.vocab_size, size=self.seq_len + 1, dtype=np.int64)
        x = torch.from_numpy(toks[:-1])
        y = torch.from_numpy(toks[1:])
        return x, y


class BinTokenDataset(Dataset):
    def __init__(self, path: str, seq_len: int, dtype=np.uint32):
        self.data = np.memmap(path, dtype=dtype, mode="r")
        self.seq_len = seq_len
        self.n = (len(self.data) - 1) // seq_len

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        s = idx * self.seq_len
        chunk = np.asarray(self.data[s:s + self.seq_len + 1], dtype=np.int64)
        x = torch.from_numpy(chunk[:-1])
        y = torch.from_numpy(chunk[1:])
        return x, y
