"""Tokenizer loading.

Loads a HF/SentencePiece tokenizer from a local folder or the HF Hub and wraps it
in a tiny uniform interface (`encode`, `eos_id`, `vocab_size`). The v1 tokenizer
`AkiraXan/velvet-tok-100k-unigram` is a SentencePiece unigram model; AutoTokenizer
handles it when the repo ships a `tokenizer.json`/config, with a sentencepiece
fallback for a bare `.model`.
"""

import os
import glob


class Tokenizer:
    def __init__(self, encode_fn, eos_id: int, vocab_size: int, name: str):
        self._encode = encode_fn
        self.eos_id = eos_id
        self.vocab_size = vocab_size
        self.name = name

    def encode(self, text: str):
        return self._encode(text)


def load_tokenizer(source: str, hf_token: str = None) -> Tokenizer:
    """`source` is a local dir or an HF repo id."""
    # 1) Try HF AutoTokenizer (works for local dir or hub id with tokenizer.json/config)
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(source, token=hf_token)
        eos = tok.eos_token_id
        if eos is None:
            eos = tok.pad_token_id if tok.pad_token_id is not None else tok.vocab_size - 1
        return Tokenizer(
            lambda t: tok.encode(t, add_special_tokens=False),
            int(eos), int(tok.vocab_size), f"hf:{source}",
        )
    except Exception as e:
        auto_err = e

    # 2) Fallback: a bare sentencepiece .model in a local folder
    if os.path.isdir(source):
        models = glob.glob(os.path.join(source, "*.model"))
        if models:
            import sentencepiece as spm
            sp = spm.SentencePieceProcessor(model_file=models[0])
            eos = sp.eos_id() if sp.eos_id() >= 0 else sp.get_piece_size() - 1
            return Tokenizer(
                lambda t: sp.encode(t, out_type=int),
                int(eos), int(sp.get_piece_size()), f"spm:{models[0]}",
            )
    raise RuntimeError(f"Could not load tokenizer from '{source}': {auto_err}")
