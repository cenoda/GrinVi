"""
GrinVi Core Model — Decoder-only Transformer.

Architecture highlights:
  ✦ RMSNorm (pre-norm)
  ✦ Rotary Position Embeddings (RoPE)
  ✦ Grouped-Query Attention (GQA)  ← generalises MHA & MQA
  ✦ SwiGLU Feed-Forward Network
  ✦ Gradient Checkpointing  ← trade compute for VRAM
  ✦ Optional KV-cache for fast auto-regressive inference
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as grad_checkpoint
from safetensors.torch import save_file, load_file

from grinvi.config import GrinViConfig


# ---------------------------------------------------------------------------
# RMSNorm
# ---------------------------------------------------------------------------

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return x * rms * self.weight


# ---------------------------------------------------------------------------
# Rotary Position Embeddings (RoPE)
# ---------------------------------------------------------------------------

def precompute_freqs_cis(head_dim: int, max_seq_len: int, theta: float = 10000.0) -> torch.Tensor:
    """Precompute complex-valued frequency tensor for RoPE."""
    freqs = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
    t = torch.arange(max_seq_len, device=freqs.device)
    freqs = torch.outer(t, freqs)           # (max_seq_len, head_dim/2)
    return torch.polar(torch.ones_like(freqs), freqs)  # complex64


def apply_rotary_emb(xq: torch.Tensor, xk: torch.Tensor,
                     freqs_cis: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply RoPE to query and key tensors."""
    # xq, xk: (B, seq_len, n_heads, head_dim)
    def rotate(x):
        x_ = x.float().reshape(*x.shape[:-1], -1, 2)
        x_complex = torch.view_as_complex(x_)
        return torch.view_as_real(x_complex * freqs_cis.unsqueeze(0).unsqueeze(2)).flatten(-2).type_as(x)

    return rotate(xq), rotate(xk)


# ---------------------------------------------------------------------------
# Grouped-Query Attention
# ---------------------------------------------------------------------------

class GroupedQueryAttention(nn.Module):
    def __init__(self, config: GrinViConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.n_heads = config.num_attention_heads
        self.n_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.n_rep = self.n_heads // self.n_kv_heads   # repetitions for GQA
        self.dropout_p = config.dropout

        self.q_proj = nn.Linear(self.hidden_size, self.n_heads   * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.n_heads * self.head_dim, self.hidden_size, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        freqs_cis: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        B, T, _ = hidden_states.shape

        xq = self.q_proj(hidden_states).view(B, T, self.n_heads,   self.head_dim)
        xk = self.k_proj(hidden_states).view(B, T, self.n_kv_heads, self.head_dim)
        xv = self.v_proj(hidden_states).view(B, T, self.n_kv_heads, self.head_dim)

        # Apply RoPE
        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)

        # KV-cache (used during inference)
        if kv_cache is not None:
            cache_k, cache_v = kv_cache
            xk = torch.cat([cache_k, xk], dim=1)
            xv = torch.cat([cache_v, xv], dim=1)
        new_kv_cache = (xk, xv)

        # Expand KV heads to match Q heads  (GQA → repeat interleave)
        if self.n_rep > 1:
            xk = xk.repeat_interleave(self.n_rep, dim=2)
            xv = xv.repeat_interleave(self.n_rep, dim=2)

        # (B, n_heads, seq_len, head_dim)
        xq = xq.transpose(1, 2)
        xk = xk.transpose(1, 2)
        xv = xv.transpose(1, 2)

        # Scaled dot-product attention
        dropout = self.dropout_p if self.training else 0.0
        attn_output = F.scaled_dot_product_attention(
            xq, xk, xv,
            attn_mask=attention_mask,
            dropout_p=dropout,
            is_causal=(attention_mask is None),
        )  # (B, n_heads, T, head_dim)

        attn_output = attn_output.transpose(1, 2).contiguous().view(B, T, -1)
        return self.o_proj(attn_output), new_kv_cache


# ---------------------------------------------------------------------------
# SwiGLU Feed-Forward Network
# ---------------------------------------------------------------------------

class SwiGLUFFN(nn.Module):
    """SwiGLU: out = SiLU(gate_proj(x)) * up_proj(x) → down_proj"""
    def __init__(self, config: GrinViConfig):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj   = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        self.dropout   = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x)))


# ---------------------------------------------------------------------------
# Transformer Block
# ---------------------------------------------------------------------------

class GrinViBlock(nn.Module):
    def __init__(self, config: GrinViConfig):
        super().__init__()
        self.input_layernorm    = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.self_attn          = GroupedQueryAttention(config)
        self.post_attn_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp                = SwiGLUFFN(config)
        self.dropout            = nn.Dropout(config.dropout)

    def forward(
        self,
        hidden_states: torch.Tensor,
        freqs_cis: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        # Pre-norm + Self-attention + Residual
        residual = hidden_states
        hidden_states, new_kv_cache = self.self_attn(
            self.input_layernorm(hidden_states),
            freqs_cis=freqs_cis,
            attention_mask=attention_mask,
            kv_cache=kv_cache,
        )
        hidden_states = residual + self.dropout(hidden_states)

        # Pre-norm + FFN + Residual
        residual = hidden_states
        hidden_states = residual + self.mlp(self.post_attn_layernorm(hidden_states))

        return hidden_states, new_kv_cache


# ---------------------------------------------------------------------------
# GrinVi Language Model
# ---------------------------------------------------------------------------

class GrinViModel(nn.Module):
    """
    GrinVi — decoder-only transformer language model.

    Usage:
        config = GrinViConfig.small()
        model  = GrinViModel(config)
        logits = model(input_ids)             # (B, T, vocab_size)
        loss   = model(input_ids, labels)     # scalar
    """

    def __init__(self, config: GrinViConfig):
        super().__init__()
        self.config = config
        self.gradient_checkpointing = False

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size,
                                         padding_idx=config.pad_token_id)
        self.layers = nn.ModuleList([GrinViBlock(config) for _ in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # LM head (optionally tied to embeddings)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.embed_tokens.weight

        # Precompute RoPE frequencies — stored as float, not complex buffer
        # (keeping as complex64 causes dtype-cast warnings when model.to(bfloat16) is called)
        head_dim = config.head_dim if config.head_dim is not None else config.hidden_size // config.num_attention_heads
        freqs = precompute_freqs_cis(head_dim, config.max_position_embeddings, config.rope_theta)
        # Store real and imaginary parts separately so .to(dtype) works cleanly
        self.register_buffer("freqs_cis_real", freqs.real, persistent=False)
        self.register_buffer("freqs_cis_imag", freqs.imag, persistent=False)

        # Weight initialisation
        self.apply(self._init_weights)

    def enable_gradient_checkpointing(self):
        """Trade VRAM for compute — recomputes activations on backward pass."""
        self.gradient_checkpointing = True
        print("[GrinVi] Gradient checkpointing enabled — reduces VRAM ~30-40%")

    def disable_gradient_checkpointing(self):
        self.gradient_checkpointing = False

    # ------------------------------------------------------------------
    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    # ------------------------------------------------------------------
    def forward(
        self,
        input_ids: torch.Tensor,                         # (B, T)
        labels: Optional[torch.Tensor] = None,           # (B, T) for LM loss
        attention_mask: Optional[torch.Tensor] = None,   # (B, T) pad mask (1=keep, 0=mask)
        kv_caches: Optional[list] = None,                # list of per-layer KV caches
    ):
        B, T = input_ids.shape
        assert T <= self.config.max_position_embeddings, (
            f"Sequence length {T} exceeds max_position_embeddings {self.config.max_position_embeddings}"
        )

        hidden_states = self.embed_tokens(input_ids)         # (B, T, hidden_size)

        # Retrieve the relevant slice of RoPE frequencies (reconstruct complex from real+imag parts)
        start = 0 if kv_caches is None else kv_caches[0][0].shape[1]
        freqs_cis = torch.complex(
            self.freqs_cis_real[start: start + T].float(),
            self.freqs_cis_imag[start: start + T].float(),
        )

        # Build additive attention mask from padding mask if provided
        attn_mask = None
        if attention_mask is not None:
            # Convert to additive float mask (0 = attend, -inf = ignore)
            attn_mask = (1.0 - attention_mask.float()) * torch.finfo(hidden_states.dtype).min
            attn_mask = attn_mask.unsqueeze(1).unsqueeze(1)   # (B, 1, 1, T)

        new_kv_caches = []
        for i, layer in enumerate(self.layers):
            kv_cache = kv_caches[i] if kv_caches is not None else None
            if self.gradient_checkpointing and self.training and kv_cache is None:
                # Gradient checkpointing: wrap the layer call
                # We can't checkpoint with kv_cache since it needs to return cache tensors
                def _ckpt(hs, fc, am, layer=layer):
                    out, _ = layer(hs, fc, am, None)
                    return out
                hidden_states = grad_checkpoint(_ckpt, hidden_states, freqs_cis, attn_mask, use_reentrant=False)
                new_kv_caches.append(None)
            else:
                hidden_states, new_kv = layer(hidden_states, freqs_cis, attn_mask, kv_cache)
                new_kv_caches.append(new_kv)

        hidden_states = self.norm(hidden_states)              # (B, T, hidden_size)
        logits = self.lm_head(hidden_states)                  # (B, T, vocab_size)

        loss = None
        if labels is not None:
            # Shift so that token i predicts token i+1
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1),
                ignore_index=self.config.pad_token_id,
            )
            return loss

        return logits, new_kv_caches

    # ------------------------------------------------------------------
    # Parameter count utility
    # ------------------------------------------------------------------
    def num_parameters(self, only_trainable: bool = True) -> int:
        params = self.parameters() if not only_trainable else filter(lambda p: p.requires_grad, self.parameters())
        return sum(p.numel() for p in params)

    # ------------------------------------------------------------------
    # Save / Load  (safetensors + config)
    # ------------------------------------------------------------------
    def save_pretrained(self, path: str):
        from pathlib import Path
        p = Path(path)
        p.mkdir(parents=True, exist_ok=True)
        self.config.save(p)
        state_dict = self.state_dict()
        if self.config.tie_word_embeddings and "lm_head.weight" in state_dict:
            # Avoid saving shared tensor storage twice.
            state_dict = dict(state_dict)
            state_dict.pop("lm_head.weight")
        save_file(state_dict, str(p / "model.safetensors"))
        print(f"[GrinVi] Model saved to {p}")

    @classmethod
    def from_pretrained(cls, path: str, device: str = "cpu") -> "GrinViModel":
        from pathlib import Path
        p = Path(path)
        config = GrinViConfig.load(p)
        model = cls(config)
        state_dict = load_file(str(p / "model.safetensors"), device=device)
        model.load_state_dict(state_dict, strict=not config.tie_word_embeddings)
        if config.tie_word_embeddings:
            model.lm_head.weight = model.embed_tokens.weight
        model.to(device)
        print(f"[GrinVi] Model loaded from {p}")
        return model

