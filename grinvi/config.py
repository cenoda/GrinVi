"""
GrinVi Model Configuration.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional


@dataclass
class GrinViConfig:
    # ---------- Model dimensions ----------
    vocab_size: int = 32000          # size of the vocabulary
    hidden_size: int = 768           # embedding dimension (d_model)
    intermediate_size: int = 2048    # FFN inner dimension (SwiGLU gate proj)
    num_hidden_layers: int = 12      # number of transformer layers
    num_attention_heads: int = 12    # number of query heads
    num_key_value_heads: int = 4     # number of KV heads (GQA; set equal to num_attention_heads for MHA)
    head_dim: Optional[int] = None   # auto-computed if None

    # ---------- Context ----------
    max_position_embeddings: int = 2048   # maximum sequence length
    rope_theta: float = 10000.0           # RoPE base frequency

    # ---------- Regularisation ----------
    dropout: float = 0.0              # attention and residual dropout
    rms_norm_eps: float = 1e-6        # epsilon for RMSNorm

    # ---------- Tokenizer ----------
    bos_token_id: int = 1
    eos_token_id: int = 2
    pad_token_id: int = 0

    # ---------- Misc ----------
    tie_word_embeddings: bool = True  # tie input/output embeddings
    use_flash_attention: bool = False # use Flash-Attention kernel when available

    def __post_init__(self):
        if self.head_dim is None:
            assert self.hidden_size % self.num_attention_heads == 0, (
                f"hidden_size ({self.hidden_size}) must be divisible by "
                f"num_attention_heads ({self.num_attention_heads})"
            )
            self.head_dim = self.hidden_size // self.num_attention_heads

    # ------------------------------------------------------------------
    # Preset factory methods
    # ------------------------------------------------------------------
    @classmethod
    def tiny(cls) -> "GrinViConfig":
        """~15 M parameter model for quick experiments."""
        return cls(
            vocab_size=32000,
            hidden_size=256,
            intermediate_size=512,
            num_hidden_layers=4,
            num_attention_heads=4,
            num_key_value_heads=4,
            max_position_embeddings=512,
        )

    @classmethod
    def small(cls) -> "GrinViConfig":
        """~117 M parameter model."""
        return cls(
            vocab_size=32000,
            hidden_size=768,
            intermediate_size=2048,
            num_hidden_layers=12,
            num_attention_heads=12,
            num_key_value_heads=4,
            max_position_embeddings=2048,
        )

    @classmethod
    def medium(cls) -> "GrinViConfig":
        """~345 M parameter model."""
        return cls(
            vocab_size=32000,
            hidden_size=1024,
            intermediate_size=4096,
            num_hidden_layers=24,
            num_attention_heads=16,
            num_key_value_heads=8,
            max_position_embeddings=4096,
        )

    @classmethod
    def large(cls) -> "GrinViConfig":
        """~760 M parameter model."""
        return cls(
            vocab_size=32000,
            hidden_size=1536,
            intermediate_size=6144,
            num_hidden_layers=24,
            num_attention_heads=16,
            num_key_value_heads=8,
            max_position_embeddings=4096,
        )

    # ------------------------------------------------------------------
    # Serialisation helpers
    # ------------------------------------------------------------------
    def save(self, path: str | Path):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        with open(path / "config.json", "w") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> "GrinViConfig":
        path = Path(path)
        with open(path / "config.json") as f:
            data = json.load(f)
        return cls(**data)

    def __repr__(self):
        lines = ["GrinViConfig("]
        for k, v in asdict(self).items():
            lines.append(f"  {k}={v!r},")
        lines.append(")")
        return "\n".join(lines)

