# Research: CUDA LLM Validation

## Decision: Primary Dependencies
**Chosen**: `llama-cpp-python` (with CUDA support)

**Rationale**: llama-cpp-python is the most mature and actively maintained library for GGUF model inference with CUDA acceleration. It provides native GGUF support (no conversion needed), excellent performance with quantized models (Q4_K_M, Q5_K_M, etc.), and offers pre-built wheels with CUDA support via `CMAKE_ARGS="-DLLAMA_CUDA=on" pip install llama-cpp-python`. It has robust WSL2 compatibility and is widely adopted in the community.

**Alternatives considered**:
- **ctransformers**: Deprecated/unmaintained since mid-2023. No longer recommended for production use. Limited GGUF support.
- **transformers (Hugging Face)**: Does not natively support GGUF format. Requires model conversion to PyTorch/safetensors format. Adds unnecessary complexity and storage overhead for quantized inference.
- **exllamav2/vLLM**: Excellent performance but don't support GGUF format (use GPTQ/EXL2/AWQ instead).

## Decision: CUDA Detection Approach
**Chosen**: Layered approach using `torch.cuda` + `pynvml` (nvidia-ml-py3)

**Rationale**: PyTorch's `torch.cuda` provides reliable CUDA availability checking and basic GPU queries (`is_available()`, `get_device_name()`, `get_device_capability()`). For detailed GPU metrics (VRAM, driver version, temperature, utilization), `pynvml` (Python bindings to NVIDIA Management Library) is the industry standard. This combination provides both high-level compatibility checking and low-level hardware introspection without requiring pycuda's complexity.

**Alternatives considered**:
- **pycuda alone**: More complex setup, requires CUDA toolkit compilation. Overkill for detection/monitoring.
- **llama-cpp-python's built-in checks**: Limited GPU info exposure, harder to query before library initialization.
- **subprocess calls to nvidia-smi**: Fragile, harder to parse, no structured API.

## Decision: Model Download Strategy
**Chosen**: `huggingface_hub` library with `hf_hub_download()`

**Rationale**: The official `huggingface_hub` library is the standard tool for downloading models from Hugging Face. It handles authentication, caching (stores in `~/.cache/huggingface/hub`), resume capability, and integrates seamlessly with `tqdm` for progress bars. For GGUF files, use `hf_hub_download(repo_id, filename)` to download specific quantized files (e.g., "mistral-7b-instruct-v0.2.Q4_K_M.gguf") without downloading entire repos.

**Alternatives considered**:
- **git lfs clone**: Downloads entire repository including multiple quantization variants. Wasteful for single-file needs.
- **Direct wget/curl**: No caching, no resume, no progress tracking, must manually handle auth tokens.
- **transformers.AutoModel**: Doesn't support GGUF files.

## Decision: Performance Monitoring
**Chosen**: `pynvml` for GPU metrics + Python `time` module for inference timing

**Rationale**: `pynvml` provides real-time GPU memory usage via `nvmlDeviceGetMemoryInfo()`, power draw, temperature, and utilization. For inference metrics, use Python's `time.perf_counter()` to measure time-to-first-token (TTFT) and total generation time. Calculate tokens/second from generation length and elapsed time. This lightweight approach avoids heavy profiling frameworks while providing validation-critical metrics.

**Alternatives considered**:
- **torch.cuda.max_memory_allocated()**: Only works if PyTorch tensors are involved. llama-cpp-python manages memory internally.
- **nvidia-smi in background**: Polling overhead, harder to synchronize with inference events.
- **CUDA profiling tools (nvprof, Nsight)**: Excessive complexity for simple validation checks.

## Implementation Notes

### Installation Requirements
- **CUDA Toolkit**: Must be installed on system (CUDA 11.8+ or 12.x recommended for WSL2)
- **llama-cpp-python with CUDA**: Requires build-time flags:
  ```bash
  CMAKE_ARGS="-DLLAMA_CUDA=on" pip install llama-cpp-python
  ```
  Or use pre-built wheels: `pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu121`

### Key Gotchas
- **WSL2 CUDA drivers**: Ensure NVIDIA WSL2 GPU driver is installed (Windows side), not Linux CUDA toolkit
- **Model file sizes**: Mistral 7B Q4_K_M is ~4.1GB. Ensure sufficient disk space and download timeout handling
- **GPU memory**: 7B Q4_K_M requires ~5-6GB VRAM for inference. Check available VRAM before loading
- **Context length**: Default context can consume additional VRAM. For validation, use minimal context (512-1024 tokens)
- **Thread configuration**: llama-cpp-python defaults may not be optimal. Set `n_threads` based on CPU cores, `n_gpu_layers` to offload all layers to GPU

### Python Dependencies (pip-installable)
```
llama-cpp-python  # With CUDA support
huggingface-hub   # Model downloads
nvidia-ml-py3     # GPU monitoring (pynvml)
torch             # For torch.cuda detection (or use cuda-python as lightweight alternative)
tqdm              # Progress bars
```

### Recommended Test Model
- **Repository**: `TheBloke/Mistral-7B-Instruct-v0.2-GGUF`
- **File**: `mistral-7b-instruct-v0.2.Q4_K_M.gguf` (~4.1GB)
- **Why**: Widely tested, good quality, reasonable size, active maintenance

### Validation Flow
1. Check `torch.cuda.is_available()` - fail fast if no CUDA
2. Query GPU info with `pynvml` - verify compute capability ≥ 6.0
3. Check available VRAM - ensure ≥ 6GB free
4. Download GGUF model with `hf_hub_download()` - show progress
5. Load model with `llama-cpp-python.Llama(model_path, n_gpu_layers=-1)` - measure load time
6. Run test inference - measure TTFT and tokens/sec
7. Monitor GPU memory during inference - verify GPU utilization

### Error Handling Priorities
- **No CUDA detected**: Clear message about driver installation
- **Insufficient VRAM**: Report required vs available
- **Model download failure**: Check network, auth token, disk space
- **Model load failure**: Check CUDA version compatibility, quantization format support
