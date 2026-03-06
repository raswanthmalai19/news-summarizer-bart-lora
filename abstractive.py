"""
Abstractive Summarization using BART + LoRA (PEFT fine-tuned) Model

LoRA (Low-Rank Adaptation) is applied on top of facebook/bart-large-cnn via the
PEFT library.  At inference time the code:
  1. Loads the base BART model.
  2. Wraps it with a LoRA PeftModel.
  3. If a fine-tuned adapter checkpoint exists at LORA_ADAPTER_PATH it is loaded
     automatically, giving you the fine-tuned behaviour.
  4. Falls back gracefully to the vanilla BART weights when no adapter is found.

To fine-tune the adapter yourself, run:
    python lora_finetune.py
"""
import logging
import os
import re

# ── Base model imports ────────────────────────────────────────────────────────
try:
    from transformers import BartForConditionalGeneration, BartTokenizer
    BART_AVAILABLE = True
except ImportError:
    BART_AVAILABLE = False
    logging.warning("Transformers library not available. Only extractive summarization will work.")

# ── PEFT / LoRA imports ───────────────────────────────────────────────────────
try:
    from peft import (
        get_peft_model,
        LoraConfig,
        TaskType,
        PeftModel,
    )
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False
    logging.warning(
        "PEFT library not available. LoRA fine-tuning disabled. "
        "Install it with: pip install peft"
    )

# ── Path where the fine-tuned LoRA adapter weights will be saved / loaded ─────
LORA_ADAPTER_PATH = os.path.join(os.path.dirname(__file__), "lora_adapter")

# ── LoRA hyper-parameters (used both here and in lora_finetune.py) ────────────
LORA_CONFIG = dict(
    task_type=TaskType.SEQ_2_SEQ_LM,    # BART is a seq-to-seq model
    r=16,                                # LoRA rank
    lora_alpha=32,                       # scaling factor  (alpha / r = 2)
    lora_dropout=0.05,
    # Target the query and value projection matrices in every attention layer.
    # These names match facebook/bart-large-cnn's internal module names.
    target_modules=["q_proj", "v_proj"],
    bias="none",
)

# ── Global singletons ─────────────────────────────────────────────────────────
bart_model = None
bart_tokenizer = None


def preprocess_text(text):
    """Clean and preprocess text"""
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def tokenize_sentences(text):
    """Split text into sentences using regex"""
    sentences = re.split(r'(?<=[.!?])\s+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    return sentences


def _build_lora_model(base_model):
    """
    Wrap *base_model* with LoRA adapter weights.

    Priority order:
      1. Load saved fine-tuned adapter from LORA_ADAPTER_PATH   (best quality)
      2. Attach a fresh (untrained) LoRA adapter                 (fallback)
      3. Return the raw base model if PEFT is not installed       (safe fallback)
    """
    if not PEFT_AVAILABLE:
        logging.info("PEFT not installed – using vanilla BART without LoRA.")
        return base_model

    # ── Option 1: load a previously trained adapter ───────────────────────────
    if os.path.isdir(LORA_ADAPTER_PATH):
        try:
            model = PeftModel.from_pretrained(base_model, LORA_ADAPTER_PATH)
            model.eval()
            logging.info(
                "LoRA adapter loaded from '%s'. Fine-tuned BART is active.",
                LORA_ADAPTER_PATH,
            )
            return model
        except Exception as exc:
            logging.warning(
                "Could not load LoRA adapter from '%s': %s. "
                "Falling back to fresh LoRA adapter.",
                LORA_ADAPTER_PATH,
                exc,
            )

    # ── Option 2: attach a fresh (untrained) LoRA adapter ────────────────────
    try:
        cfg = LoraConfig(**LORA_CONFIG)
        model = get_peft_model(base_model, cfg)
        model.eval()
        trainable, total = model.get_nb_trainable_parameters()
        logging.info(
            "Fresh LoRA adapter attached. Trainable params: %d / %d (%.2f%%). "
            "Run lora_finetune.py to train.",
            trainable,
            total,
            100 * trainable / total,
        )
        return model
    except Exception as exc:
        logging.warning(
            "Could not attach LoRA adapter: %s. Using vanilla BART.", exc
        )
        return base_model


def initialize_bart_model():
    """
    Load the BART base model, then wrap it with the LoRA adapter.
    Results are cached in the module-level globals so the model is only
    loaded once per process lifetime.
    """
    global bart_model, bart_tokenizer

    if not BART_AVAILABLE:
        return False

    try:
        if bart_model is None:
            logging.info(
                "Loading BART base model (facebook/bart-large-cnn)… "
                "This may take a few minutes on the first run."
            )
            bart_tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")
            base = BartForConditionalGeneration.from_pretrained(
                "facebook/bart-large-cnn"
            )
            bart_model = _build_lora_model(base)
            logging.info("BART + LoRA model ready.")
        return True
    except Exception as exc:
        logging.error("Failed to load BART model: %s", exc)
        return False


def summarize_abstractive(text, max_length=300, min_length=100):
    """
    Generate an abstractive summary using the LoRA-enhanced BART model.

    Parameters
    ----------
    text        : str  – The source article text.
    max_length  : int  – Maximum token length of the generated summary.
    min_length  : int  – Minimum token length of the generated summary.

    Returns
    -------
    str – The generated summary, or an error message.
    """
    global bart_model, bart_tokenizer

    if not BART_AVAILABLE:
        return (
            "Abstractive summarization not available. "
            "Install dependencies: pip install transformers torch peft"
        )

    if not initialize_bart_model():
        return "Failed to load BART + LoRA model for abstractive summarization."

    try:
        # ── Pre-process ───────────────────────────────────────────────────────
        text = preprocess_text(text)
        if len(text) > 2000:
            text = text[:2000]

        # ── Tokenise ──────────────────────────────────────────────────────────
        inputs = bart_tokenizer(
            text,
            return_tensors="pt",
            max_length=1024,
            truncation=True,
            padding=True,
        )

        # ── Generate ──────────────────────────────────────────────────────────
        # PeftModel forwards attribute access to the wrapped model automatically,
        # so .generate() works correctly with beam search.
        summary_ids = bart_model.generate(
            inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_length=max_length,
            min_length=min_length,
            num_beams=6,
            no_repeat_ngram_size=3,
            length_penalty=1.0,
            early_stopping=True,
            do_sample=True,
            temperature=0.8,
        )

        summary = bart_tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        return summary.strip()

    except Exception as exc:
        logging.error("Error in abstractive summarization: %s", exc)
        return f"Error generating abstractive summary: {exc}"