"""
grinvi/tokenizer_sp.py — SentencePiece-based tokenizer for multilingual/Korean support.

Advantages over tiktoken cl100k_base:
  ✓ Handles CJK languages (Korean, Chinese, Japanese) efficiently
  ✓ Can be trained on any language corpus
  ✓ Compatible with any model size
  ✓ Open-source and lightweight

Example usage:
    # Use a pre-trained Korean model from HuggingFace
    tok = GrinViTokenizerSP.from_pretrained("kyujinpy/KoAlpaca-KoTokenizer")

    # Or train your own on Korean text
    tok = GrinViTokenizerSP.train("data/korean_corpus.txt", vocab_size=32000)
    tok.save("my_korean_tokenizer.model")
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional, Union

import sentencepiece as spm
import torch


class GrinViTokenizerSP:
    """
    SentencePiece-based tokenizer with special tokens for BOS/EOS/PAD.
    Works great for Korean and other non-Latin scripts.
    """

    def __init__(self, model_path: str, bos_id: int = 1, eos_id: int = 2, pad_id: int = 0):
        """
        Args:
            model_path: Path to a .model SentencePiece model file
            bos_id: Beginning-of-sequence token ID
            eos_id: End-of-sequence token ID
            pad_id: Padding token ID
        """
        self.sp = spm.SentencePieceProcessor()
        self.sp.Load(model_path)
        self.bos_token_id = bos_id
        self.eos_token_id = eos_id
        self.pad_token_id = pad_id

    # ------------------------------------------------------------------
    @property
    def vocab_size(self) -> int:
        """Total vocabulary size (including special tokens)."""
        return self.sp.vocab_size()

    # ------------------------------------------------------------------
    def encode(
        self,
        text: str,
        add_bos: bool = True,
        add_eos: bool = True,
    ) -> List[int]:
        """Tokenise text to token IDs."""
        ids = self.sp.EncodeAsIds(text)
        if add_bos:
            ids = [self.bos_token_id] + ids
        if add_eos:
            ids = ids + [self.eos_token_id]
        return ids

    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        """Decode token IDs back to text."""
        if skip_special_tokens:
            special = {self.bos_token_id, self.eos_token_id, self.pad_token_id}
            token_ids = [t for t in token_ids if t not in special]
        return self.sp.DecodeIds(token_ids)

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
        """Encode a batch of strings."""
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

    # ------------------------------------------------------------------
    @classmethod
    def train(
        cls,
        texts: Union[str, List[str]],
        output_prefix: str = "grinvi_sp",
        vocab_size: int = 32000,
        character_coverage: float = 0.9995,
        model_type: str = "bpe",  # or "unigram", "char", "word"
    ) -> "GrinViTokenizerSP":
        """
        Train a SentencePiece model on a corpus.

        Args:
            texts: Path to text file or list of text strings
            output_prefix: Output model will be saved as {output_prefix}.model
            vocab_size: Size of the vocabulary
            character_coverage: For CJK, 0.9995 is recommended
            model_type: SentencePiece algorithm (bpe is default)

        Returns:
            Trained tokenizer instance
        """
        if isinstance(texts, list):
            # Write temporary file if given a list
            tmp_file = f"{output_prefix}_tmp.txt"
            with open(tmp_file, "w", encoding="utf-8") as f:
                for text in texts:
                    f.write(text + "\n")
            input_file = tmp_file
        else:
            input_file = texts

        print(f"[GrinVi] Training SentencePiece tokenizer on {input_file} …")
        print(f"  vocab_size: {vocab_size}")
        print(f"  model_type: {model_type}")
        print(f"  character_coverage: {character_coverage}")

        spm.SentencePieceTrainer.train(
            input=input_file,
            model_prefix=output_prefix,
            vocab_size=vocab_size,
            model_type=model_type,
            character_coverage=character_coverage,
            unk_surface=r"<unk>",
            normalization_rule_name="identity",  # preserve case
        )

        print(f"[GrinVi] Tokenizer saved to {output_prefix}.model")

        # Clean up temporary file if we created one
        if isinstance(texts, list):
            os.remove(tmp_file)

        return cls(f"{output_prefix}.model")

    # ------------------------------------------------------------------
    @classmethod
    def from_pretrained(cls, model_name_or_path: str) -> "GrinViTokenizerSP":
        """
        Load a pre-trained SentencePiece model from HuggingFace or local path.

        Examples:
            tok = GrinViTokenizerSP.from_pretrained("kyujinpy/KoAlpaca-KoTokenizer")
            tok = GrinViTokenizerSP.from_pretrained("./path/to/local.model")
        """
        model_path = model_name_or_path
        if "/" in model_name_or_path and not model_name_or_path.startswith("/"):
            # HuggingFace model ID
            from huggingface_hub import hf_hub_download
            print(f"[GrinVi] Downloading {model_name_or_path} from HuggingFace …")
            model_path = hf_hub_download(
                repo_id=model_name_or_path,
                filename="tokenizer.model",
                cache_dir="./hf_cache/",
            )
        return cls(model_path)

    def save(self, output_path: str):
        """Copy the model file to a new location."""
        import shutil
        shutil.copy(self.sp.model_file(), output_path)
        print(f"[GrinVi] Tokenizer saved to {output_path}")

