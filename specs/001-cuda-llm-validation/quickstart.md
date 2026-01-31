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

### 1. Install uv (if not already installed)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2. Clone Repository

```bash
git clone https://github.com/kenhia/kcuda.git
cd kcuda
```

### 3. Install with uv

```bash
# Install project and all dependencies (creates venv automatically)
uv sync
```

### 4. Verify Installation

```bash
uv run kcuda-validate --version
```

Expected output showing version and dependencies.

## Quick Validation

Let's validate your setup step by step.

### Step 1: Detect GPU

First, verify CUDA is available and your GPU is detected:

```bash
uv run kcuda-validate detect
```

**Expected output:**
```
GPU Detection Results

CUDA Status
  CUDA Available: ✓ Yes
  CUDA Version: 12.1

GPU Information
  Name: NVIDIA GeForce RTX 3060
  VRAM Total: 12,288 MB
  VRAM Free: 11,456 MB
  Compute Capability: 8.6

GPU Detection - PASSED
```

**If this fails**, check your NVIDIA driver and WSL2 GPU access with `nvidia-smi`.

### Step 2: Load Model

Download and load a model into GPU memory:

```bash
uv run kcuda-validate load
```

This uses the default model (Mistral 7B Q4_K_M). **First run downloads ~4GB** which may take 5-30 minutes depending on your connection.

**Expected output:**
```
Downloading model from Hugging Face...
Repository: Ttimofeyka/MistralRP-Noromaid-NSFW-Mistral-7B-GGUF
File: mistralrp-noromaid-nsfw-mistral-7b.Q4_K_M.gguf
✓ Downloaded to: ~/.cache/huggingface/hub/...

Loading model into memory...
✓ Loading...

Model Information:
  Repository: Ttimofeyka/MistralRP-Noromaid-NSFW-Mistral-7B-GGUF
  Filename: mistralrp-noromaid-nsfw-mistral-7b.Q4_K_M.gguf
  File Size: 4,168 MB
  Parameters: 7.95B
  Quantization: Q4_K_M
  Context Length: 8,192 tokens
  VRAM Usage: 4,793 MB
  Status: Loaded

✓ Model loaded successfully - PASSED!
```

**Subsequent runs** skip the download and load in ~10 seconds.

### Step 3: Run Inference

Test that the model can generate text:

```bash
uv run kcuda-validate infer "Write a haiku about coding"
```

**Expected output:**
```
Inference Test

Input: Write a haiku about coding

Generated Response:
  Code flows like water,
  Bugs emerge from shadows deep,
  Debug lights the way.

Performance Metrics:
  Tokens Generated: 23
  Time Taken: 1.45s
  Tokens/Second: 15.86
  First Token Latency: 0.23s

Inference Test - PASSED
```

### Step 4: Full Validation (Optional)

Once you've verified each step works, you can use the all-in-one command:

```bash
uv run kcuda-validate validate-all
```

This runs all three steps automatically and shows a summary:

```
→ Step 1/3: GPU Detection
✓ CUDA Available: Yes
✓ GPU Detected: NVIDIA GeForce RTX 3060
  
→ Step 2/3: Model Loading
✓ Model loaded: 4,793 MB VRAM used

→ Step 3/3: Inference Test
✓ Inference completed: 15.86 tokens/second

═══════════════════════════════════════════
Validation Summary: SUCCESS
System ready for LLM development.
═══════════════════════════════════════════
```

## Individual Commands

### GPU Detection Only

```bash
uv run kcuda-validate detect
```

Use this to verify CUDA availability without downloading models.

### Load Specific Model

```bash
uv run kcuda-validate load \
  --repo-id "Ttimofeyka/MistralRP-Noromaid-NSFW-Mistral-7B-GGUF" \
  --filename "mistralrp-noromaid-nsfw-mistral-7b.Q4_K_M.gguf"
```

### Run Custom Inference

```bash
uv run kcuda-validate infer "Write a short poem about code"
```

Or with options:

```bash
uv run kcuda-validate infer \
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
uv run kcuda-validate load \
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
