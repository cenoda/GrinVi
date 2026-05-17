"""
GrinVi Text Generator — auto-regressive decoding with multiple strategies.

Strategies
----------
  greedy        — always pick the highest-probability token
  top-k         — sample from the top-k tokens
  top-p         — nucleus sampling (top-p)
  temperature   — scale logits before sampling
"""
from __future__ import annotations

from typing import Any, List, Optional

import torch
import torch.nn.functional as F

from grinvi.model import GrinViModel
class Generator:
    def __init__(
        self,
        model: GrinViModel,
        tokenizer: Any,
        device: Optional[str] = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device or next(model.parameters()).device

    # ------------------------------------------------------------------
    @torch.inference_mode()
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 200,
        temperature: float = 1.0,
        top_k: Optional[int] = 50,
        top_p: Optional[float] = 0.9,
        repetition_penalty: float = 1.0,
        do_sample: bool = True,
        stop_on_eos: bool = True,
        min_new_tokens: int = 0,
        ban_tokens: Optional[List[int]] = None,
    ) -> str:
        """
        Generate text from a string prompt and return the decoded output
        (prompt + completion).
        """
        input_ids = self.tokenizer.encode(prompt, add_bos=True, add_eos=False)
        generated = self._generate_ids(
            torch.tensor([input_ids], dtype=torch.long, device=self.device),
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            do_sample=do_sample,
            stop_on_eos=stop_on_eos,
            min_new_tokens=min_new_tokens,
            ban_tokens=ban_tokens,
        )
        return self.tokenizer.decode(generated[0].tolist())

    # ------------------------------------------------------------------
    @torch.inference_mode()
    def _generate_ids(
        self,
        input_ids: torch.Tensor,          # (1, T)
        max_new_tokens: int = 200,
        temperature: float = 1.0,
        top_k: Optional[int] = 50,
        top_p: Optional[float] = 0.9,
        repetition_penalty: float = 1.0,
        do_sample: bool = True,
        stop_on_eos: bool = True,
        min_new_tokens: int = 0,
        ban_tokens: Optional[List[int]] = None,
    ) -> torch.Tensor:
        self.model.eval()
        kv_caches = None
        full_ids = input_ids.clone()
        prompt_len = input_ids.size(1)
        eos_id = getattr(self.tokenizer, "eos_token_id", None)
        ban_set = set(ban_tokens or [])

        for step in range(max_new_tokens):
            # Use KV-cache: only pass the last token after the first step
            if kv_caches is None:
                ctx = full_ids
            else:
                ctx = full_ids[:, -1:]

            logits, kv_caches = self.model(ctx, kv_caches=kv_caches)
            next_logits = logits[:, -1, :]  # (1, vocab_size)

            # Hard ban: certain token IDs are always disallowed
            if ban_set:
                for tid in ban_set:
                    next_logits[0, tid] = float("-inf")

            # min_new_tokens: suppress EOS until we've generated at least N new tokens
            generated_so_far = full_ids.size(1) - prompt_len
            if eos_id is not None and generated_so_far < min_new_tokens:
                next_logits[0, eos_id] = float("-inf")

            # Repetition penalty
            if repetition_penalty != 1.0:
                for tok in full_ids[0].tolist():
                    if next_logits[0, tok] < 0:
                        next_logits[0, tok] *= repetition_penalty
                    else:
                        next_logits[0, tok] /= repetition_penalty

            # Temperature
            if temperature != 1.0:
                next_logits = next_logits / temperature

            if do_sample:
                # Top-k filtering
                if top_k is not None and top_k > 0:
                    kth_vals = torch.topk(next_logits, min(top_k, next_logits.size(-1)))[0][:, -1:]
                    next_logits = next_logits.masked_fill(next_logits < kth_vals, float("-inf"))

                # Top-p (nucleus) filtering
                if top_p is not None and top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(next_logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    remove_mask = cumulative_probs - F.softmax(sorted_logits, dim=-1) > top_p
                    sorted_logits = sorted_logits.masked_fill(remove_mask, float("-inf"))
                    next_logits = sorted_logits.scatter(1, sorted_indices, sorted_logits)

                probs = F.softmax(next_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                # Greedy
                next_token = next_logits.argmax(dim=-1, keepdim=True)

            full_ids = torch.cat([full_ids, next_token], dim=1)

            if stop_on_eos and next_token.item() == self.tokenizer.eos_token_id:
                break

        return full_ids

    # ------------------------------------------------------------------
    def stream(
        self,
        prompt: str,
        max_new_tokens: int = 200,
        temperature: float = 1.0,
        top_k: Optional[int] = 50,
        top_p: Optional[float] = 0.9,
    ):
        """
        Generator that yields decoded tokens one-by-one (streaming mode).

        Example::

            for token in gen.stream("Once upon a time"):
                print(token, end="", flush=True)
        """
        input_ids = self.tokenizer.encode(prompt, add_bos=True, add_eos=False)
        ids_tensor = torch.tensor([input_ids], dtype=torch.long, device=self.device)
        kv_caches = None
        full_ids = ids_tensor.clone()

        self.model.eval()
        with torch.inference_mode():
            for _ in range(max_new_tokens):
                ctx = full_ids if kv_caches is None else full_ids[:, -1:]
                logits, kv_caches = self.model(ctx, kv_caches=kv_caches)
                next_logits = logits[:, -1, :] / temperature
                next_logits_top = next_logits.clone()
                if top_k:
                    kth = torch.topk(next_logits_top, min(top_k, next_logits_top.size(-1)))[0][:, -1:]
                    next_logits_top = next_logits_top.masked_fill(next_logits_top < kth, float("-inf"))
                probs = F.softmax(next_logits_top, dim=-1)
                next_token = torch.multinomial(probs, 1)
                full_ids = torch.cat([full_ids, next_token], dim=1)
                token_str = self.tokenizer.decode([next_token.item()], skip_special_tokens=False)
                if next_token.item() == self.tokenizer.eos_token_id:
                    break
                yield token_str

