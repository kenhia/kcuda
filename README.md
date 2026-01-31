# CUDA LLM Hardware Validation

A minimal Python CLI tool to validate NVIDIA GPU hardware accessibility in WSL2 for LLM inference using CUDA acceleration.

## Quick Start

See the [Quickstart Guide](specs/001-cuda-llm-validation/quickstart.md) for installation and usage.

## Overview

This tool validates that your WSL2 environment can:
- ✅ Detect NVIDIA GPU hardware via CUDA
- ✅ Download and load GGUF model files into GPU memory
- ✅ Execute GPU-accelerated inference for text generation

## Installation

### Prerequisites

1. Install [uv](https://github.com/astral-sh/uv) if not already installed:
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

2. Ensure NVIDIA drivers are installed on Windows host (for WSL2)

### Install from Source

```bash
# Clone repository
git clone https://github.com/kenhia/kcuda.git
cd kcuda

# Install with uv (creates virtual environment automatically)
uv sync

# Run the tool
uv run kcuda-validate --version
```

### Verify Installation

```bash
# Check all dependencies are installed
uv run kcuda-validate --version

# Run GPU detection to verify CUDA setup
uv run kcuda-validate detect
```

## Usage

### Quick Validation

Run the complete validation pipeline in one command:

```bash
uv run kcuda-validate validate-all
```

This will:
1. Detect your NVIDIA GPU and verify CUDA availability
2. Download a test model (first run only, ~4GB)
3. Load the model into GPU memory
4. Run a simple inference test

### Individual Commands

```bash
# Detect GPU hardware only
uv run kcuda-validate detect

# Load a specific model
uv run kcuda-validate load \
  --repo-id TheBloke/Mistral-7B-Instruct-v0.2-GGUF \
  --filename mistral-7b-instruct-v0.2.Q4_K_M.gguf

# Run inference with custom prompt
uv run kcuda-validate infer "Explain quantum computing in simple terms"

# Run inference with options
uv run kcuda-validate infer \
  --max-tokens 100 \
  --temperature 0.8 \
  "Tell me a story"
```

### Global Options

```bash
# Verbose output (DEBUG logging)
uv run kcuda-validate -v detect

# Quiet mode (ERROR only)
uv run kcuda-validate -q validate-all

# Custom log file
uv run kcuda-validate --log-file /tmp/kcuda.log detect

# Show version and dependencies
uv run kcuda-validate --version
```

### Environment Variables

```bash
# Custom log directory
export KCUDA_LOG_DIR=~/logs
uv run kcuda-validate detect

# Custom log level
export KCUDA_LOG_LEVEL=DEBUG
uv run kcuda-validate detect
```

## Requirements

- **WSL2** with Ubuntu 20.04+ or Debian 11+
- **NVIDIA GPU** with 6GB+ VRAM
- **NVIDIA Driver** 510.06+ on Windows host
- **Python 3.11+**
- **uv** package manager

## Documentation

- [Specification](specs/001-cuda-llm-validation/spec.md) - Feature requirements and user stories
- [Quickstart Guide](specs/001-cuda-llm-validation/quickstart.md) - Setup and troubleshooting
- [Implementation Plan](specs/001-cuda-llm-validation/plan.md) - Technical architecture
- [CLI Contracts](specs/001-cuda-llm-validation/contracts/cli.md) - Command interface specification

## Development

```bash
# Run tests
uv run pytest tests/ -v

# Format code
uv run ruff format .

# Lint code
uv run ruff check --fix .

# Pre-commit checks (constitution requirement)
uv run ruff format . && uv run ruff check --fix . && uv run pytest
```

## License

MIT
