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


# ---------------------------------------------------------------------------
# Module-level helpers for multiprocessing-based training.
# Workers each hold their own Kiwi instance via a pool initializer.
# ---------------------------------------------------------------------------
_WORKER_KIWI = None
_WORKER_INCLUDE_POS = True


def _worker_init(include_pos: bool):
    global _WORKER_KIWI, _WORKER_INCLUDE_POS
    from kiwipiepy import Kiwi
    _WORKER_KIWI = Kiwi()
    _WORKER_INCLUDE_POS = include_pos


def _worker_count_chunk(lines):
    """Process a batch of raw lines and return (morph_counter, char_counter).

    Uses **per-word** kiwi.tokenize calls — identical algorithm to the
    single-process implementation so the resulting vocab is deterministic and
    bit-identical across process counts.
    """
    kiwi = _WORKER_KIWI
    include_pos = _WORKER_INCLUDE_POS
    morph_counter: Counter = Counter()
    char_counter: Counter = Counter()
    for raw_text in lines:
        text = raw_text.strip()
        if not text:
            continue
        for ch in text:
            if ch.isspace():
                continue
            if 0xAC00 <= ord(ch) <= 0xD7A3:
                continue
            char_counter[ch] += 1
        for word in text.split():
            morphs = kiwi.tokenize(word)
            if not morphs:
                piece = ("▁" + word + "/UNK") if include_pos else ("▁" + word)
                morph_counter[piece] += 1
                continue
            for index, token in enumerate(morphs):
                piece = f"{token.form}/{token.tag}" if include_pos else token.form
                if index == 0:
                    piece = "▁" + piece
                morph_counter[piece] += 1
    return morph_counter, char_counter


class GrinViMorphTokenizer:
    """Kiwi-based morphology tokenizer for Korean text."""

    PAD_TOKEN = "<|pad|>"
    BOS_TOKEN = "<|bos|>"
    EOS_TOKEN = "<|eos|>"
    UNK_TOKEN = "<|unk|>"

    # Character-level fallback markers. Any morpheme that misses the vocab is
    # split into single-character pieces of the form ``<c:X>`` (or
    # ``▁<c:X>`` when at the start of a whitespace-delimited word). This makes
    # the tokenizer effectively zero-UNK as long as the per-character vocab
    # covers the alphabet of the input.
    CHAR_OPEN = "<c:"
    CHAR_CLOSE = ">"

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

    def __getstate__(self):
        state = self.__dict__.copy()
        if "kiwi" in state:
            del state["kiwi"]
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        from kiwipiepy import Kiwi
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

    def _char_piece(self, ch: str, is_word_start: bool) -> str:
        piece = f"{self.CHAR_OPEN}{ch}{self.CHAR_CLOSE}"
        return ("▁" + piece) if is_word_start else piece

    def _piece_is_char(self, piece: str) -> bool:
        body = piece[1:] if piece.startswith("▁") else piece
        return body.startswith(self.CHAR_OPEN) and body.endswith(self.CHAR_CLOSE)

    def _char_of_piece(self, piece: str) -> str:
        body = piece[1:] if piece.startswith("▁") else piece
        return body[len(self.CHAR_OPEN):-len(self.CHAR_CLOSE)]

    def encode(
        self,
        text: str,
        add_bos: bool = True,
        add_eos: bool = True,
    ) -> List[int]:
        pieces = self._tokenize_to_pieces(text)
        ids: List[int] = []
        for piece in pieces:
            tid = self.token_to_id.get(piece)
            if tid is not None:
                ids.append(tid)
                continue
            # Character-level fallback: split the surface form into single
            # characters and look up `<c:X>` pieces. Preserves the `▁`
            # word-start marker on the first character.
            is_word_start = piece.startswith("▁")
            body = piece[1:] if is_word_start else piece
            if self.include_pos and "/" in body:
                form, _tag = body.rsplit("/", 1)
            else:
                form = body
            if not form:
                ids.append(self.unk_token_id)
                continue
            for i, ch in enumerate(form):
                ch_piece = self._char_piece(ch, is_word_start=(i == 0 and is_word_start))
                ids.append(self.token_to_id.get(ch_piece, self.unk_token_id))
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

        words: List[str] = []
        current_morphs: List[tuple] = []
        current_chars: List[str] = []  # raw chars accumulated for the current word

        def flush_word():
            """Emit the currently-built word. If any char-level pieces were
            used, prefer raw concatenation over kiwi.join (kiwi would mangle
            arbitrary chars). Otherwise reconstruct via kiwi.join."""
            if not current_morphs and not current_chars:
                return ""
            if current_chars and not current_morphs:
                joined = "".join(current_chars)
            elif current_chars and current_morphs:
                # Mixed (rare): concat chars then morph surface forms.
                joined = "".join(current_chars) + "".join(f for f, _ in current_morphs)
            else:
                try:
                    joined = self.kiwi.join(current_morphs)
                except Exception:
                    joined = "".join(form for form, tag in current_morphs)
            current_morphs.clear()
            current_chars.clear()
            return joined

        for token_id in token_ids:
            if skip_special_tokens and token_id in special_ids:
                continue

            token = self.id_to_token[token_id] if 0 <= token_id < len(self.id_to_token) else self.UNK_TOKEN

            if token == self.UNK_TOKEN:
                w = flush_word()
                if w: words.append(w)
                words.append("<unk>")
                continue

            is_word_start = token.startswith("▁")

            # Character-level piece — append raw char to current word.
            if self._piece_is_char(token):
                if is_word_start:
                    w = flush_word()
                    if w: words.append(w)
                current_chars.append(self._char_of_piece(token))
                continue

            surface_token = token[1:] if is_word_start else token

            # When a new word starts, flush the previous morphemes
            if is_word_start:
                w = flush_word()
                if w: words.append(w)

            if self.include_pos and "/" in surface_token:
                form, tag = surface_token.rsplit("/", 1)
                current_morphs.append((form, tag))
            else:
                current_morphs.append((surface_token, "UNK"))

        # Flush the last word
        w = flush_word()
        if w: words.append(w)

        return " ".join(words)

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
        extra_char_top_n: int = 4000,
        num_workers: Optional[int] = None,
        chunk_size: int = 10000,
    ) -> "GrinViMorphTokenizer":
        """Train a morph tokenizer with character-level fallback.

        Parallelized over ``num_workers`` processes (default: all CPU cores).
        The result is deterministic and bit-identical to the single-process
        version because Counter aggregation is order-independent and the
        final sort uses a total ordering on (count, token).
        """
        try:
            from kiwipiepy import Kiwi  # noqa: F401  (checked here for early error)
        except ImportError as exc:
            raise ImportError(
                "kiwipiepy is required for GrinViMorphTokenizer. Install it with 'pip install kiwipiepy'."
            ) from exc

        import os
        import time
        import multiprocessing as mp

        if num_workers is None:
            num_workers = max(1, (os.cpu_count() or 4) - 1)

        morph_counter: Counter[str] = Counter()
        char_counter: Counter[str] = Counter()

        # Build a generator that yields lists of lines (chunks) so workers
        # process meaningful batches instead of one line at a time.
        def iter_chunks():
            buf: List[str] = []
            if isinstance(texts, list):
                source = iter(texts)
            else:
                source = open(texts, "r", encoding="utf-8", errors="ignore")
            try:
                for line in source:
                    buf.append(line)
                    if len(buf) >= chunk_size:
                        yield buf
                        buf = []
                if buf:
                    yield buf
            finally:
                if not isinstance(texts, list):
                    source.close()

        # Best-effort progress estimation against file size.
        total_bytes: Optional[int] = None
        if isinstance(texts, str):
            try:
                total_bytes = Path(texts).stat().st_size
            except OSError:
                total_bytes = None

        print(
            f"[GrinVi] Training tokenizer with {num_workers} workers, "
            f"chunk_size={chunk_size}…"
        )
        t0 = time.time()
        chunks_done = 0
        bytes_seen = 0

        ctx = mp.get_context("fork")
        with ctx.Pool(
            processes=num_workers,
            initializer=_worker_init,
            initargs=(include_pos,),
        ) as pool:
            for partial_morph, partial_char in pool.imap_unordered(
                _worker_count_chunk, iter_chunks(), chunksize=1
            ):
                morph_counter.update(partial_morph)
                char_counter.update(partial_char)
                chunks_done += 1
                # Estimate bytes processed from the partial counters' total weight.
                # (Not exact but good enough for ETA.)
                if chunks_done % 5 == 0:
                    elapsed = time.time() - t0
                    # Use distinct morphemes accumulated as a rough proxy.
                    print(
                        f"[GrinVi]   chunks={chunks_done} "
                        f"morph_vocab={len(morph_counter):,} "
                        f"elapsed={elapsed:.0f}s",
                        flush=True,
                    )

        print(f"[GrinVi] Counting done in {time.time() - t0:.1f}s. "
              f"Distinct morphemes={len(morph_counter):,}, "
              f"distinct chars={len(char_counter):,}")

        special_tokens = [
            cls.PAD_TOKEN,
            cls.BOS_TOKEN,
            cls.EOS_TOKEN,
            cls.UNK_TOKEN,
        ]

        # Character pieces: Korean syllables (always) + ASCII printable (always)
        # + top-N other chars from the corpus (hanja, full-width punct, …).
        char_pieces: List[str] = []
        seen_chars: set = set()
        def _add_char(ch: str):
            if ch in seen_chars:
                return
            seen_chars.add(ch)
            char_pieces.append(f"{cls.CHAR_OPEN}{ch}{cls.CHAR_CLOSE}")
            char_pieces.append(f"▁{cls.CHAR_OPEN}{ch}{cls.CHAR_CLOSE}")
        # Korean syllables — both word-initial and middle forms.
        for cp in range(0xAC00, 0xD7A4):
            _add_char(chr(cp))
        # ASCII printable (a-z, A-Z, 0-9, punctuation) — always included.
        import string as _string
        for ch in _string.printable:
            if not ch.isspace():
                _add_char(ch)
        # Other chars by corpus frequency.
        extra_added = 0
        for ch, _cnt in char_counter.most_common():
            if extra_added >= extra_char_top_n:
                break
            if ch in seen_chars:
                continue
            _add_char(ch)
            extra_added += 1
        # No hard cap — full Korean block + ASCII + extras is the design.

        # Remaining budget for morphemes.
        max_morph = max(0, vocab_size - len(special_tokens) - len(char_pieces))
        learned_tokens = [
            token for token, _ in sorted(morph_counter.items(), key=lambda item: (-item[1], item[0]))[:max_morph]
        ]
        # Dedup against char pieces (in case a morph happens to coincide).
        char_set = set(char_pieces)
        learned_tokens = [t for t in learned_tokens if t not in char_set]
        vocab = special_tokens + char_pieces + learned_tokens

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
        print(f"  specials: {len(special_tokens)}, char pieces: {len(char_pieces)}, morphs: {len(learned_tokens)}")
        print(f"  total vocab: {len(vocab)}  (requested {vocab_size})")
        print(f"[GrinVi] Morph vocab saved to {vocab_path}")
        return cls(str(model_path))