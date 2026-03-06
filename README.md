# News Summarizer — BART + LoRA

A Flask web application that summarizes news articles using two complementary approaches:

- **Extractive summarization** — picks and ranks the most important sentences from the original article using TF-IDF scoring, word frequency analysis, and sentence position weighting. No GPU required.
- **Abstractive summarization** — generates a brand-new, fluent summary using `facebook/bart-large-cnn`, optionally enhanced with a fine-tuned LoRA (Low-Rank Adaptation) adapter via the PEFT library.

Input can be either raw pasted text or a news article URL (content is scraped automatically).

---

## Table of Contents

- [Features](#features)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Installation](#installation)
- [Running the App](#running-the-app)
- [Fine-Tuning with LoRA](#fine-tuning-with-lora)
- [API Reference](#api-reference)
- [How It Works](#how-it-works)
- [Tech Stack](#tech-stack)

---

## Features

- Paste any news article text or drop in a URL to scrape the content automatically
- Choose between **extractive** (fast, no GPU needed) and **abstractive** (higher quality, uses BART)
- Three summary length settings: **short**, **medium**, **long**
- LoRA fine-tuning script included — train on CNN/DailyMail with ~0.5 % of BART's parameters
- Graceful fallback: if PEFT or Transformers is not installed, the app still works with extractive summarization
- Clean, single-page web UI served by Flask

---

## Project Structure

```
news-summarizer-bart-lora/
├── main.py              # Flask app entry point — routes & URL scraper
├── app.py               # Alternate app module (same routes, standalone)
├── extractive.py        # TF-IDF + frequency + position sentence scorer
├── abstractive.py       # BART + LoRA inference (LORA_CONFIG shared constant)
├── lora_finetune.py     # LoRA fine-tuning script (CNN/DailyMail dataset)
├── requirements.txt     # Python dependencies
├── templates/
│   └── index.html       # Single-page web UI
└── .gitignore
```

> `lora_adapter/` (generated after fine-tuning) is git-ignored because it contains large binary weight files. Run `lora_finetune.py` to produce it locally.

---

## Requirements

| Dependency | Version | Purpose |
|---|---|---|
| Python | >= 3.9 | Runtime |
| Flask | 3.0.0 | Web server |
| requests | 2.31.0 | URL fetching |
| beautifulsoup4 | 4.12.2 | HTML scraping |
| torch | >= 2.0.0 | PyTorch backend for BART |
| transformers | >= 4.20.0 | BART model & tokenizer |
| peft | latest | LoRA adapter (optional) |
| datasets | latest | CNN/DailyMail for fine-tuning (optional) |

Minimum hardware for abstractive summarization:
- **CPU** — works, but slow (~30–90 s per summary)
- **GPU (CUDA, >= 8 GB VRAM)** — recommended for fast inference and fine-tuning

---

## Installation

```bash
# 1. Clone the repository
git clone https://github.com/raswanthmalai19/news-summarizer-bart-lora.git
cd news-summarizer-bart-lora

# 2. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate          # macOS / Linux
# .venv\Scripts\activate           # Windows

# 3. Install core dependencies
pip install -r requirements.txt

# 4. (Optional) Install LoRA / fine-tuning dependencies
pip install peft datasets
```

---

## Running the App

```bash
python main.py
```

Open your browser at **http://127.0.0.1:5000**.

On the first abstractive summarization request the app downloads `facebook/bart-large-cnn` (~1.6 GB) and caches it under `~/.cache/huggingface/`. Subsequent starts load from cache instantly.

---

## Fine-Tuning with LoRA

The `lora_finetune.py` script fine-tunes BART on the CNN/DailyMail dataset. Only ~2 million of BART's 400 million parameters are trained, so it fits on a single consumer GPU.

```bash
# Basic run (5 000 training examples, 2 epochs)
python lora_finetune.py

# Custom run
python lora_finetune.py \
    --epochs 3 \
    --batch_size 4 \
    --grad_accum 4 \
    --max_train_samples 10000 \
    --fp16                        # add this flag if you have a CUDA GPU
```

| Argument | Default | Description |
|---|---|---|
| `--epochs` | 2 | Number of training epochs |
| `--batch_size` | 2 | Per-device batch size |
| `--grad_accum` | 8 | Gradient accumulation steps (effective batch = batch × accum) |
| `--lr` | 3e-4 | Learning rate |
| `--max_train_samples` | 5000 | Cap on training examples |
| `--max_eval_samples` | 500 | Cap on validation examples |
| `--output_dir` | `./lora_adapter` | Where to save the adapter weights |
| `--fp16` | off | Enable mixed-precision (CUDA only) |

After training, adapter weights are saved to `./lora_adapter/` (~10–20 MB). Restart the Flask server and `abstractive.py` will automatically detect and load them.

---

## API Reference

### `POST /summarize`

**Request body (JSON):**

```json
{
  "text": "Full article text here...",
  "type": "extractive",
  "length": "medium"
}
```

| Field | Values | Default |
|---|---|---|
| `text` | Any string | required |
| `type` | `"extractive"` \| `"abstractive"` | `"extractive"` |
| `length` | `"short"` \| `"medium"` \| `"long"` | `"medium"` |

**Response (JSON):**

```json
{
  "summary": "Generated summary text...",
  "original_sentences": 24,
  "summary_sentences": 3,
  "type": "extractive"
}
```

### `POST /summarize_url`

Same request/response shape, but pass a `"url"` field instead of `"text"`. The server fetches and scrapes the article automatically.

---

## How It Works

### Extractive Pipeline (`extractive.py`)

```
Raw text
  → clean whitespace
  → split into sentences
  → tokenize words, remove stopwords
  → compute TF-IDF scores per word
  → score each sentence:
        40% TF-IDF  +  30% word frequency  +  20% position  +  10% length
  → pick top-N sentences
  → return in original document order
```

Position scoring gives a bonus to the opening sentences (journalists front-load key facts) and the closing sentence.

### Abstractive Pipeline (`abstractive.py`)

```
Raw text
  → normalize whitespace
  → truncate to 2 000 chars (fast pre-filter)
  → BartTokenizer  →  input_ids  [1 × ≤1024 tokens]
  → BART Encoder   →  contextual representations
  → BART Decoder   →  beam search (6 beams, no-repeat-ngram=3)
  → decode token IDs  →  summary string
```

If a fine-tuned adapter exists in `./lora_adapter/`, `PeftModel.from_pretrained()` loads it on top of the frozen base model before inference.

### LoRA Explained

Instead of updating all 400 M BART parameters, LoRA injects small trainable matrices $A \in \mathbb{R}^{d \times r}$ and $B \in \mathbb{R}^{r \times d}$ alongside the frozen query and value weight matrices in every attention layer:

$$W' = W_0 + \frac{\alpha}{r} \cdot BA$$

With rank $r = 16$ and $\alpha = 32$, only ~2 M parameters are trained — roughly **0.5 %** of the model — while the base weights stay frozen, preventing catastrophic forgetting.

---

## Tech Stack

- **Flask** — lightweight Python web framework
- **PyTorch** — tensor computation and model inference
- **Hugging Face Transformers** — `facebook/bart-large-cnn` pretrained model
- **Hugging Face PEFT** — LoRA fine-tuning and adapter loading
- **Hugging Face Datasets** — CNN/DailyMail dataset for fine-tuning
- **BeautifulSoup4** — HTML parsing for URL scraping
- **Python standard library** — `re`, `math`, `collections.Counter` for the extractive scorer (zero extra dependencies)

---

## License

MIT
