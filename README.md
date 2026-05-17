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
├── scripts/            # Training, data prep, and utility scripts
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
python scripts/train_pipeline.py \
    --data data/processed/train.txt \
    --tokenizer morph \
    --tokenizer_model data/raw/ko_wikipedia/ko_tokenizer.json \
    --checkpoint_dir checkpoints/my_experiment \
    --preset small
```

This ensures everything is configured correctly before starting an expensive run. Detailed failure modes and recovery notes are documented in [`TRAINING_RUNBOOK.md`](TRAINING_RUNBOOK.md).

### 2. Manual Training Steps (Advanced)

If you prefer to run steps manually:

#### 2.1. Smoke-test training (tiny model, synthetic data)

```bash
python scripts/train.py --preset tiny --max_steps 500 --batch_size 4
```

#### 2.2. Train on your own text

```bash
python scripts/train.py \
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
python scripts/generate.py --checkpoint checkpoints/step-final

# Single prompt
python scripts/generate.py \
    --checkpoint checkpoints/step-final \
    --prompt "Once upon a time" \
    --max_new_tokens 300 \
    --temperature 0.8 \
    --stream
```

---

## Model Presets

| Preset | Layers | Hidden | Heads | ~Params |
|--------|--------|--------|-------|---------|
| `tiny`   | 4  | 256  | 4  | ~15 M  |
| `small`  | 12 | 768  | 12 | ~117 M |
| `medium` | 24 | 1024 | 16 | ~345 M |
| `large`  | 24 | 1536 | 16 | ~760 M |

---

## Python API

```python
from grinvi import GrinViConfig, GrinViModel, GrinViTokenizer, Generator

# Build model
config = GrinViConfig.small()
model  = GrinViModel(config)
print(f"Parameters: {model.num_parameters():,}")

# Training forward pass
import torch
input_ids = torch.randint(0, config.vocab_size, (2, 128))
labels    = input_ids.clone()
loss = model(input_ids, labels=labels)

# Inference
tokenizer = GrinViTokenizer()
gen = Generator(model, tokenizer)
print(gen.generate("The meaning of life is"))

# Save / load
model.save_pretrained("my_checkpoint")
model2 = GrinViModel.from_pretrained("my_checkpoint")
```

---

## License

See [LICENSE](LICENSE).
