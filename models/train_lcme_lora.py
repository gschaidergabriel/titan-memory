#!/usr/bin/env python3
"""
LCME LoRA Training — Qwen 2.5 3B (RTX 5070 / CUDA)
=====================================================
Trains a standalone LoRA adapter that teaches any Qwen 2.5 3B model
to use LCME (Local Cognitive Memory Engine) as a tool.

No Frank personality. No project-specific stuff. Just tool-calling.

Based on train_v7_gpu.py from the Frank IAPT training suite.

Usage:
  python train_lcme_lora.py
  python train_lcme_lora.py --lr 1e-4 --epochs 3
"""

import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path

import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, TaskType

# ── Defaults ──
LOCAL_MODEL = "C:/Users/Win11/models/Qwen2.5-3B-Instruct-abliterated-hf"
HF_MODEL = "huihui-ai/Qwen2.5-3B-Instruct-abliterated"
BASE_MODEL = LOCAL_MODEL if os.path.isdir(LOCAL_MODEL) else HF_MODEL

SCRIPT_DIR = Path(__file__).parent
OUTPUT_DIR = str(SCRIPT_DIR / "lcme_lora_output")
TRAIN_FILE = str(SCRIPT_DIR / "lcme_train.jsonl")
EVAL_FILE = str(SCRIPT_DIR / "lcme_eval.jsonl")

# LoRA config — focused on tool-calling behavior
LORA_R = 16          # Lower than Frank (32) — simpler task
LORA_ALPHA = 32      # 2x rank
LORA_DROPOUT = 0.05  # Light dropout
TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",  # Full attention
    "gate_proj", "up_proj", "down_proj",       # MLP for content generation
]

# Training — RTX 5070 8GB
LEARNING_RATE = 1e-4
LR_SCHEDULER = "cosine"
WARMUP_RATIO = 0.05
NUM_EPOCHS = 3
BATCH_SIZE = 1
GRADIENT_ACCUMULATION = 8  # Effective batch = 8
MAX_SEQ_LENGTH = 512       # LCME conversations are shorter than Frank's
WEIGHT_DECAY = 0.01


def parse_args():
    p = argparse.ArgumentParser(description="LCME LoRA Training")
    p.add_argument("--base-model", type=str, default=BASE_MODEL)
    p.add_argument("--output-dir", type=str, default=OUTPUT_DIR)
    p.add_argument("--train-file", type=str, default=TRAIN_FILE)
    p.add_argument("--eval-file", type=str, default=EVAL_FILE)
    p.add_argument("--lora-r", type=int, default=LORA_R)
    p.add_argument("--lora-alpha", type=int, default=LORA_ALPHA)
    p.add_argument("--lr", type=float, default=LEARNING_RATE)
    p.add_argument("--epochs", type=int, default=NUM_EPOCHS)
    p.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    p.add_argument("--grad-accum", type=int, default=GRADIENT_ACCUMULATION)
    p.add_argument("--max-seq-len", type=int, default=MAX_SEQ_LENGTH)
    return p.parse_args()


def load_data(path, tokenizer, max_len):
    """Load JSONL and tokenize for causal LM training."""
    samples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line.strip())
            samples.append(obj)

    def tokenize(example):
        text = tokenizer.apply_chat_template(
            example["messages"], tokenize=False, add_generation_prompt=False
        )
        tokens = tokenizer(
            text, truncation=True, max_length=max_len,
            padding=False, return_tensors=None,
        )
        tokens["labels"] = tokens["input_ids"].copy()
        return tokens

    dataset = Dataset.from_list(samples)
    tokenized = dataset.map(tokenize, remove_columns=["messages"],
                            num_proc=1, desc="Tokenizing")
    return tokenized


def main():
    args = parse_args()

    print("=" * 60)
    print("LCME LoRA Training")
    print("=" * 60)
    print(f"Model:      {args.base_model}")
    print(f"Train:      {args.train_file}")
    print(f"Eval:       {args.eval_file}")
    print(f"Output:     {args.output_dir}")
    print(f"LoRA:       r={args.lora_r}, alpha={args.lora_alpha}")
    print(f"Training:   lr={args.lr}, epochs={args.epochs}, batch={args.batch_size}x{args.grad_accum}")
    print(f"Max seq:    {args.max_seq_len}")
    print(f"GPU:        {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print(f"VRAM:       {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB" if torch.cuda.is_available() else "")
    print("=" * 60)

    # Load tokenizer
    print("\n[1/5] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load data
    print("[2/5] Loading and tokenizing data...")
    train_data = load_data(args.train_file, tokenizer, args.max_seq_len)
    eval_data = load_data(args.eval_file, tokenizer, args.max_seq_len)
    print(f"  Train: {len(train_data)} samples")
    print(f"  Eval:  {len(eval_data)} samples")

    # Token length stats
    lengths = [len(x["input_ids"]) for x in train_data]
    print(f"  Token lengths: min={min(lengths)}, max={max(lengths)}, avg={sum(lengths)/len(lengths):.0f}")

    # Load model
    print("[3/5] Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )

    # Apply LoRA
    print("[4/5] Applying LoRA...")
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=LORA_DROPOUT,
        target_modules=TARGET_MODULES,
        task_type=TaskType.CAUSAL_LM,
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Training
    print("[5/5] Training...")
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type=LR_SCHEDULER,
        warmup_ratio=WARMUP_RATIO,
        weight_decay=WEIGHT_DECAY,
        fp16=True,
        bf16=False,
        logging_steps=5,
        eval_strategy="steps",
        eval_steps=50,
        save_strategy="steps",
        save_steps=50,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        report_to="none",
        dataloader_pin_memory=True,
        max_grad_norm=1.0,
        remove_unused_columns=False,
    )

    collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer, padding=True, pad_to_multiple_of=8,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=eval_data,
        data_collator=collator,
    )

    t0 = time.time()
    trainer.train()
    elapsed = time.time() - t0
    print(f"\nTraining complete in {elapsed/60:.1f} minutes")

    # Save
    final_dir = os.path.join(args.output_dir, "final")
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    print(f"Adapter saved to: {final_dir}")

    # Eval loss
    metrics = trainer.evaluate()
    print(f"Final eval loss: {metrics['eval_loss']:.4f}")

    # Cleanup
    del model, trainer
    gc.collect()
    torch.cuda.empty_cache()

    print("\nDone! Next steps:")
    print(f"  1. Convert to GGUF: python convert_lora_to_gguf.py {final_dir}")
    print(f"  2. Test: llama-server --model Qwen2.5-3B.gguf --lora lcme-lora.gguf")
    print(f"  3. Publish to LCME repo: models/lcme-lora-qwen3b.gguf")


if __name__ == "__main__":
    main()
