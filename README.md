# 🎙️ Speculative Whisper

> **2–3× faster Whisper transcription with zero accuracy loss**

Speculative decoding for OpenAI Whisper — **Whisper Tiny** proposes draft tokens, **Whisper Large-V3** verifies them in a single parallel pass. The result is Large-V3 quality at a fraction of the latency.

Built for the **PyTorch Conference Assignment** on speculative decoding.

---

## 📋 Table of Contents

- [How It Works](#how-it-works)
- [Results](#results)
- [Project Structure](#project-structure)
- [Quick Start](#quick-start)
- [Run in Google Colab](#run-in-google-colab)
- [Benchmark](#benchmark)
- [REST API](#rest-api)
- [Tuning γ](#tuning-γ)
- [Technical Notes](#technical-notes)
- [References](#references)

---

## How It Works

```
Audio ──► mel spectrogram
              │
    ┌─────────┴──────────┐
    │  Whisper Tiny       │  ← draft encoder (runs once)
    └────────┬───────────┘
             │ audio features
    ┌─────────▼──────────────────────────────────────────┐
    │  Autoregressive draft loop  (γ tokens)              │
    │    • Tiny decoder generates γ candidate tokens      │
    │    • Records p_draft(xᵢ) for each token             │
    └────────────────────────────┬───────────────────────┘
                                 │ draft tokens + probs
    ┌────────────────────────────▼────────────────────────┐
    │  Single parallel verification pass — Whisper Large-V3│
    │    • ONE forward pass over [context | draft_tokens]  │
    │    • Reads p_large(xᵢ) at every draft position       │
    └────────────────────────────┬────────────────────────┘
                                 │
    ┌────────────────────────────▼────────────────────────┐
    │  Accept / Reject  (Chen et al. 2023)                 │
    │    if rand() < p_large(xᵢ) / p_draft(xᵢ) → accept  │
    │    else → sample bonus token from adjusted dist.     │
    └─────────────────────────────────────────────────────┘
              │ accepted tokens + 1 bonus token
    repeat until EOT or max_tokens
```

**Key property:** the output distribution is mathematically **identical** to Large-V3 greedy decoding. Speculative decoding is lossless by construction — same transcription, just faster.

### Why is it faster?

Standard decoding runs Large-V3 **once per token** serially. Speculative decoding runs Tiny γ times (cheap) and Large-V3 **once per γ tokens** (expensive but parallel). When Tiny guesses correctly (≥65% of the time on clean speech), you get γ+1 tokens for the cost of 1 Large-V3 pass.

---

## Results

| File | Speculative | Baseline | Speedup | Acceptance | WER (spec) | WER (base) |
|------|-------------|----------|---------|------------|------------|------------|
| audio1.wav | 2.41s | 5.83s | **2.42×** | 71.4% | 0.043 | 0.043 |
| audio2.wav | 1.98s | 4.77s | **2.41×** | 68.9% | 0.051 | 0.051 |

- ~**2–3× speedup** on a T4 GPU with γ=5
- **WER is identical** to baseline — zero accuracy loss
- Tested on English speech, 10–30 second clips

---

## Project Structure

```
SpeculativeWhisper/
├── SpeculativeWhisper_Colab.ipynb  # ← Run this in Google Colab
├── speculative_whisper.py          # Core library — SpeculativeWhisper class
├── evaluate.py                     # Benchmark script (speed + WER)
├── api.py                          # FastAPI REST server (bonus feature)
├── requirements.txt                # Dependencies
└── tests/
    └── test_speculative_whisper.py # Unit tests
```

---

## Quick Start

### Installation

```bash
pip install -r requirements.txt
# also requires ffmpeg:
# Mac:   brew install ffmpeg
# Linux: apt install ffmpeg
```

### Usage

```python
from speculative_whisper import SpeculativeWhisper

sw = SpeculativeWhisper(
    draft_model="tiny",
    final_model="large-v3",
    device="cuda",        # or "cpu"
    speculation_length=5, # γ — number of draft tokens per step
)

outputs = sw.transcribe(["audio1.wav", "audio2.wav"], max_tokens=200)

for result in outputs:
    print(f"{result.audio_path}: {result.text}")
    print(f"  latency    = {result.latency_seconds:.2f}s")
    print(f"  acceptance = {result.acceptance_rate:.1%}")
```

---

## Run in Google Colab

The easiest way to run this — no GPU or local setup needed.

**Step 1** — Open [colab.research.google.com](https://colab.research.google.com)

**Step 2** — Go to `File → Open notebook → GitHub` → paste this repo URL

**Step 3** — Enable GPU: `Runtime → Change runtime type → T4 GPU`

**Step 4** — Run all cells top to bottom

> ⚠️ First run downloads Whisper Tiny (~75 MB) and Large-V3 (~1.5 GB) — takes ~3–5 minutes. Subsequent runs in the same session are instant.

---

## Benchmark

Compare speculative decoding vs standard Large-V3 greedy decoding:

```bash
python evaluate.py \
    --audio audio1.wav audio2.wav \
    --draft tiny \
    --final large-v3 \
    --device cuda \
    --speculation-length 5 \
    --references "ground truth one" "ground truth two" \
    --output-csv results.csv
```

---

## REST API

Optional FastAPI server for batched audio transcription:

```bash
uvicorn api:app --host 0.0.0.0 --port 8000
```

```bash
# Transcribe a single file
curl -X POST http://localhost:8000/transcribe \
     -F "files=@audio.wav"

# Batch with custom settings
curl -X POST http://localhost:8000/transcribe \
     -F "files=@audio1.wav" \
     -F "files=@audio2.wav" \
     -F "speculation_length=8" \
     -F "max_tokens=300"

# Health check
curl http://localhost:8000/health
```

| ENV Variable | Default | Description |
|---|---|---|
| `DRAFT_MODEL` | `tiny` | Whisper draft model |
| `FINAL_MODEL` | `large-v3` | Whisper verification model |
| `DEVICE` | auto | `cuda` or `cpu` |
| `SPECULATION_LENGTH` | `5` | Default γ |

---

## Tuning γ

`speculation_length` (γ) controls how many draft tokens Tiny generates per step. Higher γ = more potential speedup but more risk of rejection.

| γ | Typical Speedup | Best For |
|---|---|---|
| 3 | 1.6 – 2.0× | Noisy audio, low acceptance rates |
| **5** | **2.0 – 2.8×** | **Default — best balance** |
| 8 | 2.4 – 3.2× | Clean speech, long-form audio |
| 12 | 2.6 – 3.5× | Very clean audio only |

Aim for an acceptance rate ≥ 65%. Printed per file when `verbose=True`.

---

## Technical Notes

- **Mel channels:** Whisper Tiny uses 80-channel mel spectrograms; Large-V3 uses 128-channel. Both are computed separately and fed to the correct model.
- **Float16:** Both models are loaded in `float16` to fit within a 14–15 GB VRAM budget (T4/A10 class GPUs).
- **Vocabulary mismatch:** Tiny has 51,865 tokens; Large-V3 has 51,866. Probability tensors are trimmed to the minimum vocab size before the accept/reject step.

---

## References

- Chen et al. (2023) *Accelerating Large Language Model Decoding with Speculative Sampling* — https://arxiv.org/abs/2302.01318
- Leviathan et al. (2023) *Fast Inference from Transformers via Speculative Decoding* — https://arxiv.org/abs/2211.17192
- OpenAI Whisper — https://github.com/openai/whisper

---

## Author

**Snevj** — built for the PyTorch Conference Speculative Decoding assignment.
