"""Streaming dataset for large HF corpora (e.g. HuggingFaceFW/fineweb-edu-score-2).

Streams documents, tokenizes on the fly, inserts EOS between docs, and packs the
token stream into fixed-length (seq_len+1) windows -> (input_ids, targets). This
covers the *full* subsample without pre-tokenizing terabytes to disk, and scales
identically from the RTX 5080 to an H100 box. Worker/rank sharding keeps DataLoader
workers from yielding duplicate data.
"""

import torch
from torch.utils.data import IterableDataset, get_worker_info


class StreamingTextDataset(IterableDataset):
    def __init__(self, hf_path, tokenizer, seq_len, *, hf_name=None, split="train",
                 text_field="text", shuffle_buffer=10_000, seed=42,
                 max_doc_tokens=65_536, rank=0, world_size=1):
        self.hf_path = hf_path
        self.hf_name = hf_name
        self.split = split
        self.text_field = text_field
        self.tok = tokenizer
        self.seq_len = seq_len
        self.shuffle_buffer = shuffle_buffer
        self.seed = seed
        self.max_doc_tokens = max_doc_tokens
        self.rank = rank
        self.world_size = world_size

    def _build_stream(self, shard_index, shard_count):
        from datasets import load_dataset
        ds = load_dataset(self.hf_path, name=self.hf_name, split=self.split, streaming=True)
        if self.shuffle_buffer:
            ds = ds.shuffle(seed=self.seed, buffer_size=self.shuffle_buffer)
        if shard_count > 1:
            ds = ds.shard(num_shards=shard_count, index=shard_index)
        return ds

    def __iter__(self):
        wi = get_worker_info()
        n_workers = wi.num_workers if wi else 1
        worker_id = wi.id if wi else 0
        # combine DataLoader-worker and DDP-rank sharding into one flat shard space
        shard_count = n_workers * self.world_size
        shard_index = self.rank * n_workers + worker_id

        ds = self._build_stream(shard_index, shard_count)
        eos = self.tok.eos_id
        need = self.seq_len + 1
        buf = []
        for ex in ds:
            text = ex.get(self.text_field)
            if not text:
                continue
            ids = self.tok.encode(text)
            if self.max_doc_tokens:
                ids = ids[: self.max_doc_tokens]
            buf.append(eos)
            buf.extend(ids)
            while len(buf) >= need:
                chunk = buf[:need]
                buf = buf[self.seq_len:]          # advance by seq_len (1-token overlap for the shift)
                t = torch.tensor(chunk, dtype=torch.long)
                yield t[:-1], t[1:]
