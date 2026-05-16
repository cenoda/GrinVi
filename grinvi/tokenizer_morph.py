"""
grinvi/tokenizer_morph.py — Korean morphology-aware tokenizer built on Kiwi.

The tokenizer stores a fixed vocabulary as JSON and tokenizes text into
Korean morpheme pieces. The first morpheme of each whitespace-delimited word
is prefixed with "▁" so text can be reconstructed during decoding.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import List, Optional, Union

import torch


class GrinViMorphTokenizer:
    """Kiwi-based morphology tokenizer for Korean text."""

    PAD_TOKEN = "<|pad|>"
    BOS_TOKEN = "<|bos|>"
    EOS_TOKEN = "<|eos|>"
    UNK_TOKEN = "<|unk|>"

    def __init__(
        self,
        model_path: str,
    ):
        self.model_path = str(model_path)
        data = json.loads(Path(model_path).read_text(encoding="utf-8"))

        self.include_pos: bool = bool(data.get("include_pos", True))
        self.id_to_token: List[str] = list(data["vocab"])
        self.token_to_id = {token: idx for idx, token in enumerate(self.id_to_token)}

        self.pad_token_id = self.token_to_id[self.PAD_TOKEN]
        self.bos_token_id = self.token_to_id[self.BOS_TOKEN]
        self.eos_token_id = self.token_to_id[self.EOS_TOKEN]
        self.unk_token_id = self.token_to_id[self.UNK_TOKEN]

        try:
            from kiwipiepy import Kiwi
        except ImportError as exc:
            raise ImportError(
                "kiwipiepy is required for GrinViMorphTokenizer. Install it with 'pip install kiwipiepy'."
            ) from exc

        self.kiwi = Kiwi()

    @property
    def vocab_size(self) -> int:
        return len(self.id_to_token)

    def _piece_from_token(self, form: str, tag: str, is_word_start: bool) -> str:
        piece = f"{form}/{tag}" if self.include_pos else form
        if is_word_start:
            piece = "▁" + piece
        return piece

    def _tokenize_to_pieces(self, text: str) -> List[str]:
        pieces: List[str] = []
        for word in text.split():
            morphs = self.kiwi.tokenize(word)
            if not morphs:
                pieces.append(self._piece_from_token(word, "UNK", is_word_start=True))
                continue
            for index, token in enumerate(morphs):
                pieces.append(
                    self._piece_from_token(token.form, token.tag, is_word_start=(index == 0))
                )
        return pieces

    def encode(
        self,
        text: str,
        add_bos: bool = True,
        add_eos: bool = True,
    ) -> List[int]:
        pieces = self._tokenize_to_pieces(text)
        ids = [self.token_to_id.get(piece, self.unk_token_id) for piece in pieces]
        if add_bos:
            ids = [self.bos_token_id] + ids
        if add_eos:
            ids = ids + [self.eos_token_id]
        return ids

    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        special_ids = {
            self.pad_token_id,
            self.bos_token_id,
            self.eos_token_id,
        }
        parts: List[str] = []
        for token_id in token_ids:
            if skip_special_tokens and token_id in special_ids:
                continue
            token = self.id_to_token[token_id] if 0 <= token_id < len(self.id_to_token) else self.UNK_TOKEN
            if token == self.UNK_TOKEN:
                surface = "<unk>"
                is_word_start = bool(parts)
            else:
                is_word_start = token.startswith("▁")
                surface_token = token[1:] if is_word_start else token
                surface = surface_token.rsplit("/", 1)[0] if self.include_pos and "/" in surface_token else surface_token
            if is_word_start and parts:
                parts.append(" ")
            parts.append(surface)
        return "".join(parts)

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

    @classmethod
    def train(
        cls,
        texts: Union[str, List[str]],
        output_prefix: str = "grinvi_ko_morph",
        vocab_size: int = 64000,
        include_pos: bool = True,
    ) -> "GrinViMorphTokenizer":
        try:
            from kiwipiepy import Kiwi
        except ImportError as exc:
            raise ImportError(
                "kiwipiepy is required for GrinViMorphTokenizer. Install it with 'pip install kiwipiepy'."
            ) from exc

        kiwi = Kiwi()
        counter: Counter[str] = Counter()

        def iter_lines():
            if isinstance(texts, list):
                for text in texts:
                    yield text
            else:
                with open(texts, "r", encoding="utf-8", errors="ignore") as handle:
                    for line in handle:
                        yield line

        for raw_text in iter_lines():
            text = raw_text.strip()
            if not text:
                continue
            for word in text.split():
                morphs = kiwi.tokenize(word)
                if not morphs:
                    piece = ("▁" + word + "/UNK") if include_pos else ("▁" + word)
                    counter[piece] += 1
                    continue
                for index, token in enumerate(morphs):
                    piece = f"{token.form}/{token.tag}" if include_pos else token.form
                    if index == 0:
                        piece = "▁" + piece
                    counter[piece] += 1

        special_tokens = [
            cls.PAD_TOKEN,
            cls.BOS_TOKEN,
            cls.EOS_TOKEN,
            cls.UNK_TOKEN,
        ]
        max_vocab = max(0, vocab_size - len(special_tokens))
        learned_tokens = [
            token for token, _ in sorted(counter.items(), key=lambda item: (-item[1], item[0]))[:max_vocab]
        ]
        vocab = special_tokens + learned_tokens

        model_path = Path(f"{output_prefix}.json")
        model_path.parent.mkdir(parents=True, exist_ok=True)
        model_path.write_text(
            json.dumps(
                {
                    "type": "kiwi_morph",
                    "include_pos": include_pos,
                    "vocab": vocab,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        vocab_path = Path(f"{output_prefix}.vocab")
        vocab_path.write_text("\n".join(vocab) + "\n", encoding="utf-8")

        print(f"[GrinVi] Morph tokenizer saved to {model_path}")
        print(f"[GrinVi] Morph vocab saved to {vocab_path}")
        return cls(str(model_path))