# Quickstart: CUDA LLM Validation Tool

Get up and running with GPU-accelerated LLM validation in under 5 minutes.

## Prerequisites

- **WSL2** with Ubuntu 20.04+ or Debian 11+
- **NVIDIA GPU** with 6GB+ VRAM (8GB+ recommended)
- **NVIDIA Driver** 510.06+ installed on Windows host
- **Python 3.11+** (check with `python3 --version`)
- **uv** package manager installed

### Verify WSL2 GPU Access

Before starting, confirm your GPU is accessible:

```bash
nvidia-smi
```

You should see your GPU listed. If not, see [WSL2 GPU Setup Guide](https://docs.nvidia.com/cuda/wsl-user-guide/).

## Installation

### 1. Clone or Navigate to Repository

```bash
cd ~/src/kcuda
```

### 2. Create Virtual Environment with uv

```bash
uv venv
source .venv/bin/activate  # On Windows WSL2
```

### 3. Install Dependencies

```bash
# Install with CUDA support (pre-built wheels)
uv add llama-cpp-python \
  --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu121

# Install remaining dependencies
uv add torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
uv add pynvml huggingface-hub click rich
```

### 4. Verify Installation

```bash
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}')"
```

Expected output: `CUDA Available: True`

## Quick Validation

### Run Full Validation Pipeline

The fastest way to validate your setup:

```bash
python -m kcuda_validate validate-all
```

This will:
1. ✓ Detect your NVIDIA GPU
2. ✓ Download Mistral 7B GGUF model (~4GB, first run only)
3. ✓ Load model into GPU memory
4. ✓ Run inference test
5. ✓ Display performance metrics

**First run**: 5-30 minutes (model download)  
**Subsequent runs**: <30 seconds

### Expected Output

```
→ Step 1/3: GPU Detection
✓ CUDA Available: Yes
✓ GPU Detected: NVIDIA GeForce RTX 3060
  - VRAM Total: 12288 MB
  - CUDA Version: 12.1
  
→ Step 2/3: Model Loading
→ Downloading model (first run only)...
  [████████████████████████████] 4168 MB | 12.3 MB/s
✓ Model loaded: 4832 MB VRAM used

→ Step 3/3: Inference Test
✓ Inference completed: 13.4 tokens/second

═══════════════════════════════════════════
Validation Summary: SUCCESS
System ready for LLM development.
═══════════════════════════════════════════
```

## Individual Commands

### GPU Detection Only

```bash
python -m kcuda_validate detect
```

Use this to verify CUDA availability without downloading models.

### Load Specific Model

```bash
python -m kcuda_validate load \
  --repo-id "Ttimofeyka/MistralRP-Noromaid-NSFW-Mistral-7B-GGUF" \
  --filename "mistralrp-noromaid-nsfw-mistral-7b.Q4_K_M.gguf"
```

### Run Custom Inference

```bash
python -m kcuda_validate infer "Write a short poem about code"
```

Or with options:

```bash
python -m kcuda_validate infer \
  --max-tokens 100 \
  --temperature 0.8 \
  "Explain quantum computing simply"
```

## Common Issues

### Issue: "No CUDA-capable device detected"

**Solution**: 
1. Verify `nvidia-smi` works in WSL2
2. Check Windows NVIDIA driver version ≥ 510.06
3. Restart WSL: `wsl --shutdown` (in PowerShell), then reopen

### Issue: "Insufficient VRAM"

**Solution**: Try smaller quantization:

```bash
python -m kcuda_validate load \
  --filename "mistralrp-noromaid-nsfw-mistral-7b.Q4_K_S.gguf"  # Smaller
```

Or close GPU applications (browsers, games).

### Issue: "Model download very slow"

**Solution**: 
1. Check internet connection
2. Use `--skip-download` if model already cached
3. Manually download from Hugging Face and place in cache directory

### Issue: Import errors for llama-cpp-python

**Solution**: Reinstall with correct CUDA wheel:

```bash
uv remove llama-cpp-python
uv add llama-cpp-python \
  --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu121 \
  --no-cache
```

## Performance Benchmarks

Expected performance on common GPUs (Mistral 7B Q4_K_M):

| GPU Model          | VRAM | Tokens/Sec | Time to 1st Token |
|--------------------|------|------------|-------------------|
| RTX 4090          | 24GB | 50-60      | ~0.5s            |
| RTX 3080          | 10GB | 25-35      | ~0.7s            |
| RTX 3060          | 12GB | 15-20      | ~0.9s            |
| RTX 2060 Super    | 8GB  | 10-15      | ~1.2s            |

**Below target?** Check:
- GPU isn't thermal throttling: `nvidia-smi dmon`
- Other processes aren't using GPU
- Model fully loaded in VRAM (not paging)

## Next Steps

After validation succeeds:

1. **Explore parameters**: Try different `--temperature`, `--max-tokens`
2. **Test other models**: Change `--repo-id` to try different GGUF models
3. **Build your app**: Use this as foundation for your LLM application
4. **Profile performance**: Add `--verbose` flag for detailed timing

## Debugging

All operations are logged to file for troubleshooting:

**Default log location**: `~/.cache/kcuda/logs/kcuda-validate.log`

```bash
# View recent logs
tail -f ~/.cache/kcuda/logs/kcuda-validate.log

# Run with verbose output AND file logging
python -m kcuda_validate detect --verbose

# Specify custom log file
python -m kcuda_validate detect --log-file /tmp/debug.log
```

**Log files include**:
- CUDA driver and runtime versions
- GPU memory allocation details
- Model loading diagnostics
- Inference timing breakdown
- All errors with full stack traces

Logs are automatically rotated at 10MB with 5 backup files retained.

## Getting Help

- Check logs: Add `--verbose` to any command or review log file
- View CLI options: `python -m kcuda_validate --help`
- Model compatibility: Visit [Hugging Face GGUF models](https://huggingface.co/models?search=gguf)

## Development Mode

For development with live reloading:

```bash
# Editable install with dev dependencies
uv install -e .
kcuda-validate validate-all  # Installed as CLI command
```

Run tests:

```bash
uv run pytest tests/ -v
```

Format and lint:

```bash
uv run ruff format .
uv run ruff check --fix .
```

---

**Validation Time**: ~30 seconds (after initial model download)  
**Model Cache Location**: `~/.cache/huggingface/hub/`  
**VRAM Required**: 6GB minimum, 8GB recommended
