# CLI Contract: CUDA LLM Validation Tool

## Application Name

`kcuda-validate` (or `kcuda validate` if subcommand structure)

## Global Options

```
--verbose, -v         Enable verbose output (DEBUG level logging)
--quiet, -q           Suppress informational output (ERROR only)
--log-file PATH       Write logs to specified file [default: ~/.cache/kcuda/logs/kcuda-validate.log]
--no-log-file         Disable file logging (stdout/stderr only)
--help, -h            Show help message and exit
--version             Show version and exit
```

## Commands

### 1. detect - GPU Hardware Detection

**Purpose**: Detect and display NVIDIA GPU hardware information

**Usage**:
```bash
kcuda-validate detect [OPTIONS]
```

**Options**: None (uses global options only)

**Output Format** (success):
```
✓ CUDA Available: Yes
✓ GPU Detected: NVIDIA GeForce RTX 3060
  - VRAM Total: 12288 MB
  - VRAM Free: 11520 MB
  - CUDA Version: 12.1
  - Driver Version: 525.60.11
  - Compute Capability: 8.6
  - Device ID: 0

Hardware validation: PASSED
```

**Output Format** (failure - no GPU):
```
✗ CUDA Available: No
✗ GPU Detected: None

Error: No NVIDIA GPU detected. Ensure:
  1. NVIDIA GPU drivers are installed on Windows host (WSL2)
  2. WSL2 GPU passthrough is enabled
  3. nvidia-smi works in WSL2 terminal

Hardware validation: FAILED
```

**Exit Codes**:
- `0`: GPU detected successfully, CUDA available
- `1`: No GPU detected or CUDA unavailable
- `2`: Detection error (driver issues, permission problems)

**Success Criteria**: Maps to FR-001, FR-002, SC-001

---

### 2. load - Download and Load Model

**Purpose**: Download GGUF model from Hugging Face and load into GPU memory

**Usage**:
```bash
kcuda-validate load [OPTIONS]
```

**Options**:
```
--repo-id TEXT        Hugging Face repository ID 
                      [default: Ttimofeyka/MistralRP-Noromaid-NSFW-Mistral-7B-GGUF]
--filename TEXT       Specific GGUF file to download
                      [default: MistralRP-Noromaid-NSFW-7B-Q4_0.gguf]
--skip-download       Skip download if model already cached
--no-gpu              Load in CPU mode (for testing)
```

**Output Format** (success):
```
→ Checking model cache...
  Model: Ttimofeyka/MistralRP-Noromaid-NSFW-Mistral-7B-GGUF
  File: MistralRP-Noromaid-NSFW-7B-Q4_0.gguf

✓ Model found in cache: /home/user/.cache/huggingface/hub/models--Ttimofeyka--MistralRP-Noromaid-NSFW-Mistral-7B-GGUF/...
  Size: 4168 MB

→ Loading model into GPU memory...
  [████████████████████████████] 100%

✓ Model loaded successfully
  - Parameters: 7.24B
  - Quantization: Q4_0
  - Context Length: 8192 tokens
  - VRAM Usage: 4832 MB
  - Free VRAM Remaining: 6688 MB

Model load: PASSED
```

**Output Format** (download required):
```
→ Checking model cache...
  Model not cached. Downloading...

→ Downloading from Hugging Face:
  Ttimofeyka/MistralRP-Noromaid-NSFW-Mistral-7B-GGUF/mistralrp-noromaid-nsfw-mistral-7b.Q4_K_M.gguf
  
  [████████------] 4168/4168 MB | 12.3 MB/s | ETA: 0s

✓ Download complete
✓ Model loaded successfully
  [... same as above ...]
```

**Output Format** (failure - insufficient VRAM):
```
→ Loading model into GPU memory...

✗ Model load failed: Insufficient VRAM

Error: Model requires ~4832 MB VRAM, but only 2048 MB available.
Recommendation: Close other GPU applications or try Q4_K_S quantization (smaller).

Model load: FAILED
```

**Exit Codes**:
- `0`: Model loaded successfully
- `1`: Download failed (network, auth, file not found)
- `2`: Load failed (insufficient VRAM, corrupt file, CUDA error)
- `3`: Model validation failed (not a valid GGUF file)

**Success Criteria**: Maps to FR-003, FR-004, FR-005, FR-008, FR-010, FR-011, SC-002, SC-003

---

### 3. infer - Run Inference Test

**Purpose**: Execute text generation with loaded model to validate GPU acceleration

**Usage**:
```bash
kcuda-validate infer [OPTIONS] [PROMPT]
```

**Arguments**:
```
PROMPT                Text prompt for generation [default: "Hello, how are you?"]
```

**Options**:
```
--max-tokens INT      Maximum tokens to generate [default: 50]
--temperature FLOAT   Sampling temperature [default: 0.7]
--load-model          Automatically load default model if not loaded
--repo-id TEXT        Model repo (if --load-model used)
--filename TEXT       Model file (if --load-model used)
```

**Output Format** (success):
```
→ Checking model status...
✓ Model loaded: MistralRP-Noromaid-NSFW-7B-Q4_0.gguf

→ Running inference...
  Prompt: "Hello, how are you?"
  
─────────────────────────────────────────
Response:
Hello, how are you? I'm doing well, thank you for asking! I'm here to help you 
with any questions or tasks you might have. How can I assist you today?
─────────────────────────────────────────

✓ Inference completed successfully

Performance Metrics:
  - Tokens Generated: 43
  - Time to First Token: 0.89 seconds
  - Total Time: 3.21 seconds
  - Throughput: 13.4 tokens/second
  - GPU Utilization: 98% (peak)
  - VRAM Peak: 4912 MB

Inference test: PASSED
```

**Output Format** (failure - no model loaded):
```
→ Checking model status...
✗ No model loaded

Error: Must load model before running inference.
Run: kcuda-validate load

Or use: kcuda-validate infer --load-model "Your prompt"

Inference test: FAILED
```

**Output Format** (failure - empty prompt):
```
✗ Inference failed: Empty prompt

Error: Prompt cannot be empty. Provide text for generation.
Example: kcuda-validate infer "Tell me a story"

Inference test: FAILED
```

**Exit Codes**:
- `0`: Inference completed successfully, metrics within targets
- `1`: No model loaded (user error)
- `2`: Inference failed (CUDA error, generation error)
- `3`: Empty or invalid prompt
- `4`: Performance below targets (warning, still exits 0 if generation succeeded)

**Success Criteria**: Maps to FR-006, FR-007, FR-009, FR-012, SC-004, SC-005, SC-006, SC-007

---

### 4. validate-all - Full Validation Pipeline

**Purpose**: Run complete validation sequence (detect → load → infer) in one command

**Usage**:
```bash
kcuda-validate validate-all [OPTIONS]
```

**Options**:
```
--repo-id TEXT        Hugging Face repository ID
--filename TEXT       Specific GGUF file
--prompt TEXT         Inference test prompt [default: "Hello, how are you?"]
--skip-on-error       Continue to next step even if previous failed
```

**Output Format**: Sequential output from detect → load → infer commands

**Summary Footer**:
```
═══════════════════════════════════════════
Validation Summary:
═══════════════════════════════════════════
✓ GPU Detection: PASSED
✓ Model Loading: PASSED  
✓ Inference Test: PASSED

Overall Status: SUCCESS

System ready for LLM development.
═══════════════════════════════════════════
```

**Exit Codes**:
- `0`: All validation steps passed
- `1`: One or more validation steps failed
- `2`: Critical error prevented validation from completing

**Success Criteria**: Maps to SC-006 (end-to-end validation)

---

## Environment Variables

```
KCUDA_CACHE_DIR       Override model cache directory 
                      [default: ~/.cache/huggingface]
KCUDA_LOG_DIR         Override log file directory
                      [default: ~/.cache/kcuda/logs]
KCUDA_GPU_ID          Specify GPU device ID for multi-GPU systems
                      [default: 0]
KCUDA_LOG_LEVEL       Set log level (DEBUG, INFO, WARNING, ERROR)
                      [default: INFO]
```

## Error Message Standards

All error messages MUST follow this format:

```
✗ [Component] failed: [Brief description]

Error: [Detailed explanation of what went wrong]
Recommendation: [Actionable steps to resolve]
Log file: [Path to log file for detailed diagnostics]
[Optional: Link to documentation]

[Status]: FAILED
```

Example:
```
✗ Model load failed: Insufficient VRAM

Error: Model requires ~4832 MB VRAM, but only 2048 MB available.
Recommendation: 
  1. Close other GPU applications (browser, games)
  2. Try smaller quantization: --filename "...Q4_K_S.gguf"
  3. Check available models: huggingface.co/Ttimofeyka/...
Log file: ~/.cache/kcuda/logs/kcuda-validate.log

Model load: FAILED
```

## Contract Testing

Contract tests MUST verify:
1. All commands accept documented options
2. Exit codes match specification for each error condition
3. Output format matches examples (regex patterns)
4. Help text is complete and accurate
5. Error messages include recommendations
6. Progress indicators appear for long operations (>2 seconds)
