# How to Train MinFix (Qwen2.5-1.5B-Instruct + LoRA)

## Prerequisites

- **GPU**: NVIDIA GPU with **16+ GB VRAM** (24 GB recommended, e.g. RTX 3090/4090)
- **CUDA**: 11.8 or newer
- **Python**: 3.10 - 3.12
- **Disk**: ~10 GB free (model + checkpoints)

## Step 1: Setup Environment

```bash
# Create a virtual environment (recommended)
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Linux/Mac

# Install dependencies
cd train
pip install -r requirements.txt
```

> **Windows users**: If `bitsandbytes` fails to install, run:
> ```bash
> pip install bitsandbytes --prefer-binary
> ```
> If that still fails, use WSL2 or Linux. bitsandbytes has limited Windows support.

## Step 2: Verify GPU

```bash
python -c "import torch; print(torch.cuda.get_device_name(0)); print(f'{torch.cuda.get_device_properties(0).total_mem / 1024**3:.1f} GB VRAM')"
```

You should see your GPU name and VRAM amount.

## Step 3: Edit Config (Optional)

Open `config.yaml` to adjust training parameters. The defaults work well for a 24 GB GPU:

| Parameter | Default | Notes |
|-----------|---------|-------|
| `load_in_4bit` | `false` | Set `true` if VRAM < 16 GB (QLoRA) |
| `max_seq_length` | `2048` | Increase if diffs are long |
| `lora_r` | `64` | LoRA rank. Lower = less VRAM |
| `per_device_train_batch_size` | `2` | Reduce if OOM |
| `gradient_accumulation_steps` | `8` | Increase if batch size is reduced |

## Step 4: Start Training

```bash
cd train
python train.py
```

That's it. The script will:
1. Check your GPU and dependencies
2. Validate the dataset
3. Load the model and apply LoRA
4. Train with periodic evaluation
5. Save checkpoints and the final adapter

## Step 5: Monitor Training

Look for these in the output:
- `[VRAM]` - GPU memory usage each logging step
- `[EVAL]` - Validation loss every 50 steps
- `eval_loss` should decrease over time

### Stopping & Resuming

- **Ctrl+C** saves a checkpoint and exits gracefully
- To resume, set `resume_from_checkpoint` in `config.yaml` to the checkpoint path:
  ```yaml
  resume_from_checkpoint: "./output/checkpoint-200"
  ```

## Output

After training, find the LoRA adapter at:
```
train/output/final/
  adapter_model.safetensors   # LoRA weights
  adapter_config.json         # Adapter config
  training_info.json          # Training metadata
  tokenizer files             # Tokenizer files
```

## Troubleshooting

### Out of Memory (OOM)
1. Enable QLoRA: set `load_in_4bit: true` in config.yaml
2. Reduce `max_seq_length` (try 1024)
3. Reduce `per_device_train_batch_size` to 1
4. Increase `gradient_accumulation_steps` to 16 to keep effective batch size

### bitsandbytes install fails on Windows
Use WSL2 or install a Windows-compatible build:
```bash
pip install bitsandbytes --prefer-binary
```
If it still fails, set `load_in_4bit: false` and `bnb_4bit_*` won't be used.

### Training loss not decreasing
- Check that the dataset is loaded correctly (look for `[train] Loaded X samples`)
- Try increasing `learning_rate` to `5e-5`
- Ensure `max_seq_length` isn't too short (samples get truncated)

### Slow training
- Make sure `bf16: true` is set (requires Ampere+ GPU, RTX 30xx+)
- `gradient_checkpointing: true` saves VRAM but is slightly slower
- Increase `per_device_train_batch_size` if VRAM allows

## Advanced: Merge LoRA into Base Model

After training, merge the adapter into the full model for inference:

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct", torch_dtype="bfloat16")
model = PeftModel.from_pretrained(base, "train/output/final")
model = model.merge_and_unload()
model.save_pretrained("train/output/merged")
AutoTokenizer.from_pretrained("train/output/final").save_pretrained("train/output/merged")
```
