"""
LoRA Fine-Tuning Script for BART Abstractive Summarizer
========================================================
Fine-tunes facebook/bart-large-cnn on the CNN/DailyMail dataset using
Low-Rank Adaptation (LoRA) via the PEFT library.

Only a tiny fraction of the model's parameters are trained (~0.5 %),
so this runs on a single consumer-grade GPU (>= 8 GB VRAM) or even CPU
(slow but functional).

Usage
-----
    # Activate the project virtual environment first, then:
    python lora_finetune.py

    # Override key hyper-parameters via CLI:
    python lora_finetune.py --epochs 3 --batch_size 4 --max_train_samples 2000

After training the LoRA adapter weights are saved to ./lora_adapter/.
abstractive.py picks them up automatically on the next server start.
"""

import argparse
import logging
import os

import torch
from datasets import load_dataset
from transformers import (
    BartForConditionalGeneration,
    BartTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)
from peft import LoraConfig, get_peft_model

# ── Keep import in sync with abstractive.py ───────────────────────────────────
from abstractive import LORA_CONFIG, LORA_ADAPTER_PATH

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
BASE_MODEL_NAME = "facebook/bart-large-cnn"
DATASET_NAME    = "cnn_dailymail"
DATASET_VERSION = "3.0.0"

# Source & target field names in CNN/DailyMail
SOURCE_FIELD  = "article"
TARGET_FIELD  = "highlights"

# Token-length caps (stay within BART's 1024 limit)
MAX_SOURCE_LEN = 1024
MAX_TARGET_LEN = 128


# ─────────────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="LoRA fine-tune BART for summarisation")
    p.add_argument("--epochs",            type=int,   default=2,
                   help="Number of training epochs (default: 2)")
    p.add_argument("--batch_size",        type=int,   default=2,
                   help="Per-device training batch size (default: 2)")
    p.add_argument("--grad_accum",        type=int,   default=8,
                   help="Gradient accumulation steps (default: 8). "
                        "Effective batch = batch_size x grad_accum")
    p.add_argument("--lr",                type=float, default=3e-4,
                   help="Learning rate (default: 3e-4)")
    p.add_argument("--max_train_samples", type=int,   default=5000,
                   help="Cap on training examples (default: 5000). "
                        "Use None for the full dataset (~287k).")
    p.add_argument("--max_eval_samples",  type=int,   default=500,
                   help="Cap on validation examples (default: 500)")
    p.add_argument("--output_dir",        type=str,   default=LORA_ADAPTER_PATH,
                   help=f"Where to save adapter weights (default: {LORA_ADAPTER_PATH})")
    p.add_argument("--fp16",              action="store_true",
                   help="Enable mixed-precision training (requires CUDA)")
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
def load_and_prepare_dataset(tokenizer, max_train, max_eval):
    """Download CNN/DailyMail and tokenize it."""

    logger.info("Loading CNN/DailyMail dataset ...")
    raw = load_dataset(DATASET_NAME, DATASET_VERSION)

    # Optionally cap dataset size for quick experiments
    train_split = raw["train"]
    eval_split  = raw["validation"]
    if max_train:
        train_split = train_split.select(range(min(max_train, len(train_split))))
    if max_eval:
        eval_split = eval_split.select(range(min(max_eval, len(eval_split))))

    logger.info(
        "Dataset ready – train: %d examples, eval: %d examples",
        len(train_split), len(eval_split),
    )

    def tokenize_fn(batch):
        # Tokenize source articles
        model_inputs = tokenizer(
            batch[SOURCE_FIELD],
            max_length=MAX_SOURCE_LEN,
            truncation=True,
            padding=False,
        )
        # Tokenize target summaries as labels
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(
                batch[TARGET_FIELD],
                max_length=MAX_TARGET_LEN,
                truncation=True,
                padding=False,
            )
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    logger.info("Tokenizing dataset ...")
    tokenized_train = train_split.map(
        tokenize_fn, batched=True,
        remove_columns=train_split.column_names,
        desc="Tokenising train",
    )
    tokenized_eval = eval_split.map(
        tokenize_fn, batched=True,
        remove_columns=eval_split.column_names,
        desc="Tokenising eval",
    )

    return tokenized_train, tokenized_eval


# ─────────────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Training device: %s", device.upper())
    if device == "cpu":
        logger.warning(
            "No GPU detected.  Training on CPU will be very slow. "
            "Consider using Google Colab or a machine with a CUDA GPU."
        )

    # ── Load base model & tokenizer ───────────────────────────────────────────
    logger.info("Loading base model: %s", BASE_MODEL_NAME)
    tokenizer  = BartTokenizer.from_pretrained(BASE_MODEL_NAME)
    base_model = BartForConditionalGeneration.from_pretrained(BASE_MODEL_NAME)

    # ── Attach LoRA adapter ───────────────────────────────────────────────────
    lora_cfg = LoraConfig(**LORA_CONFIG)
    model    = get_peft_model(base_model, lora_cfg)
    model.print_trainable_parameters()          # logs trainable % to console

    # ── Dataset ───────────────────────────────────────────────────────────────
    tokenized_train, tokenized_eval = load_and_prepare_dataset(
        tokenizer,
        max_train=args.max_train_samples,
        max_eval=args.max_eval_samples,
    )

    # ── Dynamic padding collator ──────────────────────────────────────────────
    collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=-100,    # ignore padding tokens in loss
        pad_to_multiple_of=8,
    )

    # ── Training arguments ────────────────────────────────────────────────────
    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        warmup_ratio=0.05,
        weight_decay=0.01,
        fp16=args.fp16 and torch.cuda.is_available(),
        predict_with_generate=True,         # needed for validation generation
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        logging_steps=50,
        report_to="none",                   # disable wandb / tensorboard by default
        generation_max_length=MAX_TARGET_LEN,
    )

    # ── Trainer ───────────────────────────────────────────────────────────────
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_eval,
        tokenizer=tokenizer,
        data_collator=collator,
    )

    # ── Train ─────────────────────────────────────────────────────────────────
    logger.info("Starting LoRA fine-tuning ...")
    trainer.train()

    # ── Save only the LoRA adapter weights (very small, ~10-20 MB) ────────────
    os.makedirs(args.output_dir, exist_ok=True)
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    logger.info("LoRA adapter saved to '%s'", args.output_dir)
    logger.info(
        "Restart the Flask server to load the fine-tuned adapter automatically."
    )


if __name__ == "__main__":
    main()
