#!/usr/bin/env python3
"""
Run AFTER training is complete.
Merges LoRA adapter into base model and saves as safetensors.
"""

import sys
from pathlib import Path

def merge():
    try:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError:
        print("[ERROR] Missing packages. Run: pip install -r requirements.txt")
        sys.exit(1)

    script_dir = Path(__file__).parent
    final_dir = script_dir / "output" / "final"
    merged_dir = script_dir / "output" / "merged"

    if not final_dir.exists():
        print(f"[ERROR] No trained adapter found at {final_dir}")
        print("Run train.py first.")
        sys.exit(1)

    info_path = final_dir / "training_info.json"
    if info_path.exists():
        import json
        with open(info_path) as f:
            info = json.load(f)
        base_model_name = info.get("base_model", "Qwen/Qwen2.5-1.5B-Instruct")
    else:
        base_model_name = "Qwen/Qwen2.5-1.5B-Instruct"

    print(f"[1/4] Loading base model: {base_model_name}")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
        trust_remote_code=True,
    )

    print(f"[2/4] Loading LoRA adapter from: {final_dir}")
    model = PeftModel.from_pretrained(base_model, str(final_dir))

    print("[3/4] Merging adapter into base model...")
    model = model.merge_and_unload()

    merged_dir.mkdir(parents=True, exist_ok=True)

    print(f"[4/4] Saving merged model to: {merged_dir}")
    model.save_pretrained(str(merged_dir), safe_serialization=True)
    AutoTokenizer.from_pretrained(str(final_dir)).save_pretrained(str(merged_dir))

    print("\nDone! Output files:")
    for f in sorted(merged_dir.iterdir()):
        size_mb = f.stat().st_size / (1024 * 1024)
        print(f"  {f.name}  ({size_mb:.1f} MB)")

    print(f"\nLoad with:\n  AutoModelForCausalLM.from_pretrained(\"{merged_dir}\")")

if __name__ == "__main__":
    merge()
