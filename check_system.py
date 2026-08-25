#!/usr/bin/env python3
"""
Check if your system can handle training.
Run this BEFORE train.py to verify GPU, VRAM, and dependencies.
"""

import sys
import platform

def check():
    print("=" * 60)
    print("  MinFix Training - System Check")
    print("=" * 60)

    print(f"\n[Python] {sys.version}")
    print(f"[OS]     {platform.platform()}")

    try:
        import torch
        print(f"[PyTorch] {torch.__version__}")
    except ImportError:
        print("[PyTorch] NOT INSTALLED - run: pip install -r requirements.txt")
        return

    print(f"[CUDA]    available: {torch.cuda.is_available()}")
    if not torch.cuda.is_available():
        print("\n[FAIL] No CUDA GPU detected. Training requires an NVIDIA GPU.")
        return

    print(f"[CUDA]    version: {torch.version.cuda}")
    print(f"[cuDNN]   version: {torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else 'N/A'}")

    gpu_name = torch.cuda.get_device_name(0)
    props = torch.cuda.get_device_properties(0)
    vram_gb = getattr(props, 'total_memory', getattr(props, 'total_mem', 0)) / (1024 ** 3)

    print(f"\n[GPU]     {gpu_name}")
    print(f"[VRAM]    {vram_gb:.1f} GB")
    print(f"[Compute] {props.major}.{props.minor}")
    print(f"[BF16]    {torch.cuda.is_bf16_supported()}")

    print("\n" + "-" * 60)
    if vram_gb >= 20:
        print("[VERDICT] EXCELLENT - Full speed, no compromises")
        print("  Config: load_in_4bit: false, batch_size: 2+")
    elif vram_gb >= 16:
        print("[VERDICT] GOOD - Works with defaults")
        print("  Config: load_in_4bit: false, batch_size: 2")
    elif vram_gb >= 12:
        print("[VERDICT] OK - Enable QLoRA for safety")
        print("  Config: load_in_4bit: true, batch_size: 1")
    elif vram_gb >= 8:
        print("[VERDICT] TIGHT - QLoRA required")
        print("  Config: load_in_4bit: true, batch_size: 1, max_seq_length: 1024")
    else:
        print("[VERDICT] INSUFFICIENT - Need at least 8GB VRAM")
        return
    print("-" * 60)

    print("\n[Dependencies]")
    packages = {
        "torch": "torch",
        "transformers": "transformers",
        "peft": "peft",
        "datasets": "datasets",
        "trl": "trl",
        "bitsandbytes": "bitsandbytes",
        "accelerate": "accelerate",
        "yaml": "pyyaml",
    }
    all_ok = True
    for pkg, pip_name in packages.items():
        try:
            mod = __import__(pkg)
            version = getattr(mod, "__version__", "?")
            print(f"  {pip_name:20s} {version}")
        except ImportError:
            print(f"  {pip_name:20s} MISSING")
            all_ok = False

    print("\n" + "=" * 60)
    if all_ok:
        print("All checks passed. You're ready to train!")
        print("Run: python train.py")
    else:
        print("Some packages missing. Install first:")
        print("Run: pip install -r requirements.txt")
    print("=" * 60)

if __name__ == "__main__":
    check()
