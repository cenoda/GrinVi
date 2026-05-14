"""
GrinVi Tokenizer — thin wrapper around tiktoken (cl100k_base / o200k_base)
with special-token handling for BOS, EOS, and PAD.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Union

import tiktoken


class GrinViTokenizer:
    """
    Wraps tiktoken's 'cl100k_base' encoding (same as GPT-4) and adds
    BOS / EOS / PAD special tokens so the model can process sequences
    in a standard way.

    Example
    -------
    >>> tok = GrinViTokenizer()
    >>> ids = tok.encode("Hello, world!")
    >>> text = tok.decode(ids)
    """

    BOS_TOKEN = "<|bos|>"
    EOS_TOKEN = "<|eos|>"
    PAD_TOKEN = "<|pad|>"

    def __init__(self, tiktoken_model: str = "cl100k_base"):
        self._base = tiktoken.get_encoding(tiktoken_model)

        # Register special tokens on top of the base encoding
        special_tokens_dict = {
            self.BOS_TOKEN: self._base.n_vocab,
            self.EOS_TOKEN: self._base.n_vocab + 1,
            self.PAD_TOKEN: self._base.n_vocab + 2,
        }
        self._enc = tiktoken.Encoding(
            name=f"grinvi_{tiktoken_model}",
            pat_str=self._base._pat_str,
            mergeable_ranks=self._base._mergeable_ranks,
            special_tokens={**self._base._special_tokens, **special_tokens_dict},
        )

        self.bos_token_id: int = special_tokens_dict[self.BOS_TOKEN]
        self.eos_token_id: int = special_tokens_dict[self.EOS_TOKEN]
        self.pad_token_id: int = special_tokens_dict[self.PAD_TOKEN]

    # ------------------------------------------------------------------
    @property
    def vocab_size(self) -> int:
        return self._enc.n_vocab

    # ------------------------------------------------------------------
    def encode(
        self,
        text: str,
        add_bos: bool = True,
        add_eos: bool = True,
        allowed_special: str | set = "all",
    ) -> List[int]:
        ids = self._enc.encode(text, allowed_special=allowed_special)
        if add_bos:
            ids = [self.bos_token_id] + ids
        if add_eos:
            ids = ids + [self.eos_token_id]
        return ids

    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        special_ids = {self.bos_token_id, self.eos_token_id, self.pad_token_id}
        if skip_special_tokens:
            token_ids = [t for t in token_ids if t not in special_ids]
        return self._enc.decode(token_ids)

    # ------------------------------------------------------------------
    def batch_encode(
        self,
        texts: List[str],
        max_length: Optional[int] = None,
        padding: bool = True,
        truncation: bool = True,
        add_bos: bool = True,
        add_eos: bool = True,
        return_tensors: Optional[str] = None,
    ):
        """
        Encode a list of strings and optionally return PyTorch tensors.

        Returns
        -------
        dict with keys 'input_ids' and 'attention_mask'.
        """
        import torch

        encoded = [self.encode(t, add_bos=add_bos, add_eos=add_eos) for t in texts]

        if truncation and max_length is not None:
            encoded = [e[:max_length] for e in encoded]

        if padding:
            pad_len = max(len(e) for e in encoded)
            if max_length is not None:
                pad_len = min(pad_len, max_length)
            attention_masks = []
            padded = []
            for e in encoded:
                mask = [1] * len(e)
                pad_amount = pad_len - len(e)
                e_padded = e + [self.pad_token_id] * pad_amount
                mask = mask + [0] * pad_amount
                padded.append(e_padded)
                attention_masks.append(mask)
        else:
            padded = encoded
            attention_masks = [[1] * len(e) for e in encoded]

        if return_tensors == "pt":
            return {
                "input_ids": torch.tensor(padded, dtype=torch.long),
                "attention_mask": torch.tensor(attention_masks, dtype=torch.long),
            }

        return {"input_ids": padded, "attention_mask": attention_masks}

