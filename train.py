#!/usr/bin/env python3
"""
MinFix LoRA Training Script
Fine-tunes Qwen2.5-1.5B-Instruct on code-repair data using LoRA.
Designed to be robust: validates data, checks GPU, handles OOM, saves checkpoints.
"""

import os
import sys
import json
import yaml
import logging
import traceback
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

# ---------------------------------------------------------------------------
# Pre-flight: catch import errors early with helpful messages
# ---------------------------------------------------------------------------
def preflight():
    missing = []
    for pkg, imp in [
        ("torch", "torch"),
        ("transformers", "transformers"),
        ("peft", "peft"),
        ("datasets", "datasets"),
        ("trl", "trl"),
        ("bitsandbytes", "bitsandbytes"),
        ("accelerate", "accelerate"),
        ("yaml", "yaml"),
    ]:
        try:
            __import__(imp)
        except ImportError:
            missing.append(pkg)
    if missing:
        print(f"\n[ERROR] Missing packages: {', '.join(missing)}")
        print("Run:  pip install -r requirements.txt\n")
        sys.exit(1)

    import torch
    if not torch.cuda.is_available():
        print("\n[ERROR] No CUDA GPU detected. Training requires a GPU.")
        sys.exit(1)

    _props = torch.cuda.get_device_properties(0)
    vram_gb = getattr(_props, 'total_memory', getattr(_props, 'total_mem', 0)) / (1024**3)
    print(f"[INFO] GPU: {torch.cuda.get_device_name(0)} ({vram_gb:.1f} GB VRAM)")
    if vram_gb < 12:
        print("[WARN] GPU has <12 GB VRAM. Consider enabling load_in_4bit in config.yaml")

    bf16_support = torch.cuda.is_bf16_supported()
    print(f"[INFO] BF16 supported: {bf16_support}")

preflight()

import torch
from datasets import load_dataset, Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    TrainerCallback,
    set_seed,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType
from trl import SFTTrainer

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------
CONFIG_PATH = Path(__file__).parent / "config.yaml"

def load_config(path: Path = CONFIG_PATH) -> dict:
    if not path.exists():
        logger.error(f"Config not found at {path}")
        sys.exit(1)
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    logger.info(f"Loaded config from {path}")
    return cfg

# ---------------------------------------------------------------------------
# Data validation & formatting
# ---------------------------------------------------------------------------
VALID_ROLES = {"system", "user", "assistant"}

def validate_message(msg: dict, idx: int, split: str) -> bool:
    if "role" not in msg or "content" not in msg:
        logger.warning(f"[{split}] Sample {idx}: missing role or content field")
        return False
    if msg["role"] not in VALID_ROLES:
        logger.warning(f"[{split}] Sample {idx}: unknown role '{msg['role']}'")
        return False
    if not isinstance(msg["content"], str) or len(msg["content"].strip()) == 0:
        logger.warning(f"[{split}] Sample {idx}: empty content for role '{msg['role']}'")
        return False
    return True

def load_and_validate_data(path: str, split: str) -> Dataset:
    """Load JSONL, validate structure, return HF Dataset."""
    path = Path(path)
    if not path.exists():
        logger.error(f"Data file not found: {path}")
        sys.exit(1)

    samples = []
    skipped = 0
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as e:
                logger.warning(f"[{split}] Line {i}: invalid JSON - {e}")
                skipped += 1
                continue

            messages = row.get("messages")
            if not messages or not isinstance(messages, list):
                logger.warning(f"[{split}] Line {i}: missing or invalid 'messages' field")
                skipped += 1
                continue

            valid = all(validate_message(m, i, split) for m in messages)
            if not valid:
                skipped += 1
                continue

            has_assistant = any(m["role"] == "assistant" for m in messages)
            if not has_assistant:
                logger.warning(f"[{split}] Line {i}: no assistant message found")
                skipped += 1
                continue

            samples.append({"messages": messages})

    logger.info(f"[{split}] Loaded {len(samples)} samples (skipped {skipped})")
    if len(samples) == 0:
        logger.error(f"[{split}] No valid samples found! Check your data format.")
        sys.exit(1)

    return Dataset.from_list(samples)

# ---------------------------------------------------------------------------
# Chat template formatting
# ---------------------------------------------------------------------------
def format_chat(example, tokenizer, max_length=2048):
    """Apply chat template, tokenize, and truncate. Returns dict with 'input_ids' and 'attention_mask'."""
    text = tokenizer.apply_chat_template(
        example["messages"],
        tokenize=False,
        add_generation_prompt=False,
    )
    tokenized = tokenizer(
        text,
        truncation=True,
        max_length=max_length,
        padding=False,
    )
    return tokenized

# ---------------------------------------------------------------------------
# Custom callback for graceful shutdown & monitoring
# ---------------------------------------------------------------------------
class SafetyCallback(TrainerCallback):
    """Logs VRAM usage during training."""

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / (1024**3)
            reserved = torch.cuda.memory_reserved() / (1024**3)
            logger.info(f"[VRAM] Allocated: {allocated:.2f} GB | Reserved: {reserved:.2f} GB")

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics:
            loss = metrics.get("eval_loss")
            if loss is not None:
                logger.info(f"[EVAL] eval_loss = {loss:.4f}")

# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------
def train():
    cfg = load_config()

    # Set seed for reproducibility
    seed = cfg.get("seed", 42)
    set_seed(seed)

    # Resolve paths relative to this script
    script_dir = Path(__file__).parent
    train_data_path = Path(cfg["train_data"])
    val_data_path = Path(cfg["val_data"])
    if not train_data_path.is_absolute():
        train_data_path = (script_dir / train_data_path).resolve()
    if not val_data_path.is_absolute():
        val_data_path = (script_dir / val_data_path).resolve()

    output_dir = Path(cfg.get("output_dir", "./output"))
    if not output_dir.is_absolute():
        output_dir = (script_dir / output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- Tokenizer ----
    model_name = cfg["model_name"]
    logger.info(f"Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        padding_side="right",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        logger.info("Set pad_token to eos_token")

    # ---- Quantization config (QLoRA) ----
    bnb_config = None
    use_4bit = cfg.get("load_in_4bit", False)
    if use_4bit:
        logger.info("Using 4-bit quantization (QLoRA)")
        compute_dtype = torch.bfloat16 if cfg.get("bf16", True) else torch.float16
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=True,
        )

    # ---- Model ----
    logger.info(f"Loading model: {model_name}")
    model_kwargs = {
        "trust_remote_code": True,
        "dtype": torch.bfloat16 if cfg.get("bf16", True) else torch.float16,
        "device_map": "auto",
    }
    if bnb_config is not None:
        model_kwargs["quantization_config"] = bnb_config

    try:
        model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        logger.error("If you get OOM, try enabling load_in_4bit in config.yaml")
        sys.exit(1)

    if use_4bit:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

    # Print model size
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model parameters: {total_params / 1e6:.1f}M")

    # ---- LoRA ----
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=cfg.get("lora_r", 64),
        lora_alpha=cfg.get("lora_alpha", 128),
        lora_dropout=cfg.get("lora_dropout", 0.05),
        target_modules=cfg.get("lora_target_modules", [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ]),
        bias="none",
    )

    model = get_peft_model(model, lora_config)
    trainable, total = model.get_nb_trainable_parameters()
    logger.info(f"Trainable parameters: {trainable / 1e6:.1f}M / {total / 1e6:.1f}M "
                f"({100 * trainable / total:.2f}%)")

    # ---- Data ----
    logger.info("Loading and validating training data...")
    train_dataset = load_and_validate_data(str(train_data_path), "train")

    logger.info("Loading and validating validation data...")
    val_dataset = load_and_validate_data(str(val_data_path), "val")

    max_seq_length = cfg.get("max_seq_length", 2048)

    # Apply chat template + tokenize + truncate
    logger.info("Tokenizing and truncating datasets...")
    train_dataset = train_dataset.map(
        lambda ex: format_chat(ex, tokenizer, max_seq_length),
        num_proc=1,
        desc="Formatting train",
        remove_columns=train_dataset.column_names,
    )
    val_dataset = val_dataset.map(
        lambda ex: format_chat(ex, tokenizer, max_seq_length),
        num_proc=1,
        desc="Formatting val",
        remove_columns=val_dataset.column_names,
    )

    logger.info(f"Tokenized {len(train_dataset)} train, {len(val_dataset)} val samples (max_length={max_seq_length})")

    # ---- Training arguments ----
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=cfg.get("num_train_epochs", 3),
        per_device_train_batch_size=cfg.get("per_device_train_batch_size", 2),
        per_device_eval_batch_size=cfg.get("per_device_eval_batch_size", 2),
        gradient_accumulation_steps=cfg.get("gradient_accumulation_steps", 8),
        learning_rate=cfg.get("learning_rate", 2e-5),
        lr_scheduler_type=cfg.get("lr_scheduler_type", "cosine"),
        warmup_steps=cfg.get("warmup_steps", 100),
        weight_decay=cfg.get("weight_decay", 0.01),
        max_grad_norm=cfg.get("max_grad_norm", 1.0),
        fp16=cfg.get("fp16", False),
        bf16=cfg.get("bf16", True),
        optim=cfg.get("optim", "adamw_torch"),
        gradient_checkpointing=cfg.get("gradient_checkpointing", True),
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=cfg.get("logging_steps", 10),
        eval_strategy=cfg.get("eval_strategy", "steps"),
        eval_steps=cfg.get("eval_steps", 50),
        save_strategy=cfg.get("save_strategy", "steps"),
        save_steps=cfg.get("save_steps", 100),
        save_total_limit=cfg.get("save_total_limit", 3),
        load_best_model_at_end=cfg.get("load_best_model_at_end", True),
        metric_for_best_model=cfg.get("metric_for_best_model", "eval_loss"),
        greater_is_better=cfg.get("greater_is_better", False),
        report_to=cfg.get("report_to", "none"),
        seed=seed,
    )

    # ---- Callbacks ----
    callbacks = [SafetyCallback()]

    # ---- Trainer ----
    logger.info("Initializing SFTTrainer...")
    try:
        trainer = SFTTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=tokenizer,
            callbacks=callbacks,
        )
    except Exception as e:
        logger.error(f"Failed to initialize trainer: {e}")
        sys.exit(1)

    # ---- Resume from checkpoint ----
    resume_ckpt = cfg.get("resume_from_checkpoint")
    if resume_ckpt and Path(resume_ckpt).exists():
        logger.info(f"Resuming from checkpoint: {resume_ckpt}")
    else:
        resume_ckpt = None

    # ---- Train ----
    logger.info("=" * 60)
    logger.info("Starting training...")
    logger.info(f"  Epochs: {cfg.get('num_train_epochs', 3)}")
    logger.info(f"  Effective batch size: {cfg.get('per_device_train_batch_size', 2) * cfg.get('gradient_accumulation_steps', 8)}")
    logger.info(f"  Learning rate: {cfg.get('learning_rate', 2e-5)}")
    logger.info(f"  Max seq length: {max_seq_length}")
    logger.info(f"  LoRA r={cfg.get('lora_r', 64)}, alpha={cfg.get('lora_alpha', 128)}")
    logger.info(f"  Output: {output_dir}")
    logger.info("=" * 60)

    try:
        trainer.train(resume_from_checkpoint=resume_ckpt)
    except torch.cuda.OutOfMemoryError:
        logger.error("=" * 60)
        logger.error("OUT OF MEMORY!")
        logger.error("Suggestions:")
        logger.error("  1. Enable load_in_4bit in config.yaml (QLoRA)")
        logger.error("  2. Reduce max_seq_length (currently %d)", max_seq_length)
        logger.error("  3. Reduce per_device_train_batch_size")
        logger.error("  4. Increase gradient_accumulation_steps to compensate")
        logger.error("=" * 60)
        sys.exit(1)
    except Exception as e:
        logger.error(f"Training failed: {e}")
        logger.error(traceback.format_exc())
        logger.info("Partial checkpoints may be available in the output directory.")
        sys.exit(1)

    # ---- Save final model ----
    logger.info("Training complete. Saving final model...")
    try:
        final_dir = output_dir / "final"
        final_dir.mkdir(parents=True, exist_ok=True)
        trainer.model.save_pretrained(str(final_dir))
        tokenizer.save_pretrained(str(final_dir))
        logger.info(f"Model saved to {final_dir}")

        # Save adapter config summary
        adapter_config = {
            "base_model": model_name,
            "lora_r": cfg.get("lora_r", 64),
            "lora_alpha": cfg.get("lora_alpha", 128),
            "lora_dropout": cfg.get("lora_dropout", 0.05),
            "target_modules": cfg.get("lora_target_modules", []),
            "trainable_params": trainable,
            "total_params": total,
            "train_samples": len(train_dataset),
            "val_samples": len(val_dataset),
            "epochs": cfg.get("num_train_epochs", 3),
            "learning_rate": cfg.get("learning_rate", 2e-5),
            "max_seq_length": max_seq_length,
        }
        with open(final_dir / "training_info.json", "w") as f:
            json.dump(adapter_config, f, indent=2)
        logger.info("Training info saved.")
    except Exception as e:
        logger.error(f"Failed to save model: {e}")

    logger.info("Done!")

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    train()
