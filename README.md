# GrinVi 🧠
**G**eneral **R**esponse **I**ntelligence **N**euron **V**ia **I**nference

A decoder-only transformer language model built from scratch with **PyTorch**.

---

## Architecture

| Feature | Detail |
|---------|--------|
| Architecture | Decoder-only Transformer (GPT / LLaMA style) |
| Normalisation | RMSNorm (pre-norm) |
| Positional Encoding | Rotary Position Embeddings (RoPE) |
| Attention | Grouped-Query Attention (GQA) |
| Feed-Forward | SwiGLU |
| Precision | bfloat16 / float16 / float32 |
| KV-Cache | ✅ (auto-regressive inference) |

---

## Project Structure

```
GrinVi/
├── grinvi/             # Core library (Model, Tokenizer, Trainer)
├── scripts/            # Scripts categorized by function:
│   ├── data/           # Data preparation & download
│   ├── training/       # Training & pipeline automation
│   ├── monitoring/     # Diagnostics & performance
│   ├── infra/          # Vast.ai & cloud infrastructure
│   └── tools/          # Inference & demo utilities
├── tests/              # Unit tests and verification scripts
├── logs/               # Training logs and execution history
├── docs/               # Additional documentation and history
├── data/               # Local data storage (ignored by git)
├── checkpoints/        # Model checkpoints (ignored by git)
├── chat.py             # Simple interactive chat CLI
├── requirements.txt
└── pyproject.toml
```

---

## Documentation

- [**TRAINING_RUNBOOK.md**](TRAINING_RUNBOOK.md): 필수 학습 체크리스트, 트러블슈팅 가이드 및 Vast.ai 운영 팁.
- [**docs/history/**](docs/history/): 프로젝트 주요 결정 사항 및 과거 이슈 기록 (Postmortems).
- [**LICENSE**](LICENSE): Apache 2.0 License.

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 1.5. Run Training Pipeline (Recommended)

To run preflight checks, a smoke test, and then start the main training run all in one command:

```bash
python scripts/training/train_pipeline.py \
    --data data/processed/train.txt \
    --tokenizer_model data/raw/ko_wikipedia/ko_tokenizer.json \
    --checkpoint_dir checkpoints/my_experiment \
    --preset small
```

> **Note**: Default tokenizer is now `morph` and default vocab size is `80,000`.

This ensures everything is configured correctly before starting an expensive run. Detailed failure modes and recovery notes are documented in [`TRAINING_RUNBOOK.md`](TRAINING_RUNBOOK.md).

### 2. Manual Training Steps (Advanced)

If you prefer to run steps manually:

#### 2.1. Smoke-test training (tiny model, synthetic data)

```bash
python scripts/training/train.py --preset tiny --max_steps 500 --batch_size 4
```

#### 2.2. Train on your own text

```bash
python scripts/training/train.py \
    --preset small \
    --data data/train.txt \
    --eval_data data/val.txt \
    --seq_len 512 \
    --batch_size 8 \
    --grad_accum 4 \
    --max_steps 100000
```

### 3. Generate text

```bash
# Interactive REPL
python scripts/tools/inference.py --checkpoint checkpoints/step-final

# Single prompt
python scripts/tools/inference.py \
    --checkpoint checkpoints/step-final \
    --prompt "Once upon a time" \
    --max_new_tokens 300 \
    --temperature 0.8 \
    --stream
```

---

## Model Presets

| Preset | Layers | Hidden | Heads | ~Params (80k vocab) |
|--------|--------|--------|-------|---------|
| `tiny`   | 4  | 256  | 4  | ~23 M  |
| `small`  | 12 | 768  | 12 | ~137 M |
| `medium` | 24 | 1024 | 16 | ~460 M |
| `large`  | 24 | 1536 | 16 | ~972 M |

---

## Python API

```python
from grinvi import GrinViConfig, GrinViModel, GrinViMorphTokenizer, Generator

# Build model
config = GrinViConfig.small()
model  = GrinViModel(config)
print(f"Parameters: {model.num_parameters():,}")

# Training forward pass
import torch
input_ids = torch.randint(0, config.vocab_size, (2, 128))
labels    = input_ids.clone()
loss = model(input_ids, labels=labels)

# Inference (loads from checkpoint dir auto-detecting tokenizer.json)
tokenizer = GrinViMorphTokenizer.from_pretrained("my_checkpoint")
gen = Generator(model, tokenizer)
print(gen.generate("질문: 인공지능이란?\n답변:"))

# Save / load
model.save_pretrained("my_checkpoint")
model2 = GrinViModel.from_pretrained("my_checkpoint")
```

---

## License

See [LICENSE](LICENSE).
