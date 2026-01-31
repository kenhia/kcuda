# Data Model: CUDA LLM Validation

## Overview

The validation tool operates on three core entities representing hardware, model assets, and execution results. This is a read-only validation system with no persistent state beyond model file caching.

## Entities

### GPUDevice

Represents detected NVIDIA GPU hardware and its capabilities.

**Attributes**:
- `name`: GPU model name (e.g., "NVIDIA GeForce RTX 3060")
- `vram_total_mb`: Total video memory in megabytes
- `vram_free_mb`: Currently available video memory in megabytes
- `cuda_version`: CUDA runtime version string (e.g., "12.1")
- `driver_version`: NVIDIA driver version
- `compute_capability`: GPU compute capability (e.g., "8.6")
- `device_id`: Integer device index (0 for single GPU systems)

**Validation Rules**:
- `vram_total_mb` must be ≥ 4096 (4GB minimum for quantized 7B models)
- `compute_capability` must be ≥ "6.0" (Pascal or newer for efficient inference)
- `cuda_version` must be present (indicates CUDA is available)

**Lifecycle**: Created during GPU detection, immutable after creation

### LLMModel

Represents a loaded GGUF model file and its metadata.

**Attributes**:
- `repo_id`: Hugging Face repository identifier (e.g., "Ttimofeyka/MistralRP-Noromaid-NSFW-Mistral-7B-GGUF")
- `filename`: GGUF file name within repository (e.g., "mistralrp-noromaid-nsfw-mistral-7b.Q4_K_M.gguf")
- `local_path`: Absolute filesystem path to cached model file
- `file_size_mb`: Model file size in megabytes
- `parameter_count`: Number of model parameters (e.g., 7_000_000_000 for 7B)
- `quantization_type`: Quantization method (e.g., "Q4_K_M", "Q5_K_S")
- `context_length`: Maximum context window size in tokens
- `vram_usage_mb`: Measured GPU memory consumption after loading
- `is_loaded`: Boolean indicating if model is currently loaded in GPU memory

**Validation Rules**:
- `file_size_mb` must be > 0 and match expected size for quantization type
- `vram_usage_mb` must be < `GPUDevice.vram_free_mb` during load
- `quantization_type` must be valid GGUF quantization format
- `context_length` must be > 0

**Lifecycle**: 
- Created during model download/discovery
- Updated when loaded into GPU (sets `is_loaded=True`, populates `vram_usage_mb`)
- Cleaned up on application exit (memory freed)

**Relationships**:
- Requires `GPUDevice` with sufficient `vram_free_mb` to load

### InferenceResult

Represents the outcome of a single text generation request.

**Attributes**:
- `prompt`: Input text provided by user
- `response`: Generated text output from model
- `tokens_generated`: Number of tokens in generated response
- `time_to_first_token_sec`: Seconds elapsed before first token generated
- `total_time_sec`: Total generation time in seconds
- `tokens_per_second`: Calculated throughput (tokens_generated / total_time_sec)
- `gpu_utilization_percent`: GPU utilization during generation (if measurable)
- `vram_peak_mb`: Peak GPU memory usage during inference
- `success`: Boolean indicating successful completion
- `error_message`: Error details if `success=False`

**Validation Rules**:
- `prompt` must not be empty string
- `tokens_generated` must be > 0 if `success=True`
- `time_to_first_token_sec` must be ≤ `total_time_sec`
- `tokens_per_second` must be > 0 if `success=True`

**Lifecycle**: Created per inference request, immutable after generation completes

**Relationships**:
- Requires `LLMModel` with `is_loaded=True`
- Consumes GPU resources from `GPUDevice`

## State Transitions

### Model Loading State Machine

```
[Not Downloaded] 
    ↓ (download from Hugging Face)
[Downloaded/Cached]
    ↓ (load into GPU memory)
[Loaded] 
    ↓ (run inference)
[Inferencing]
    ↓ (complete/error)
[Loaded]
    ↓ (cleanup/exit)
[Unloaded]
```

**Valid Transitions**:
- Not Downloaded → Downloaded: Successful download with integrity check
- Downloaded → Loaded: Sufficient VRAM available, valid GGUF format
- Loaded → Inferencing: Valid prompt provided
- Inferencing → Loaded: Generation completes or errors
- Loaded → Unloaded: Explicit cleanup or application exit

**Invalid Transitions** (error conditions):
- Not Downloaded → Loaded: Cannot load non-existent model
- Downloaded → Inferencing: Must load into GPU first
- Unloaded → Inferencing: Cannot infer on unloaded model

## Data Flow

```
User Input (prompt)
    ↓
GPUDevice Detection → validate CUDA availability
    ↓
Model Download → LLMModel (cached)
    ↓
Model Loading → LLMModel (in GPU) + update vram_usage_mb
    ↓
Inference Request → InferenceResult
    ↓
Display Results → stdout (GPUDevice info, Model stats, InferenceResult metrics)
```

## Persistence

- **No Database**: All entities are in-memory only
- **File Caching**: Models cached by huggingface_hub in `~/.cache/huggingface/hub/`
- **No User Data**: No configuration files, user profiles, or persistent state
- **Ephemeral Results**: Inference results logged to stdout, not saved

## Memory Constraints

Based on typical GGUF quantization sizes for Mistral 7B:

| Quantization | File Size | VRAM Required | Target GPU VRAM |
|--------------|-----------|---------------|-----------------|
| Q4_K_M       | ~4.1 GB   | ~4.8 GB       | 6 GB+          |
| Q5_K_M       | ~5.0 GB   | ~5.8 GB       | 8 GB+          |
| Q8_0         | ~7.7 GB   | ~8.5 GB       | 10 GB+         |

Application will prefer Q4_K_M for maximum compatibility with consumer GPUs.
