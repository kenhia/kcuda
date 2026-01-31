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

```bash
# Create virtual environment
uv venv
source .venv/bin/activate

# Install dependencies and project in editable mode
uv add llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu121
uv add torch --index-url https://download.pytorch.org/whl/cu121
uv add pynvml huggingface-hub click rich pytest ruff --dev

# Install project in editable mode with dev dependencies
uv install -e .
```

## Usage

```bash
# Full validation pipeline
kcuda-validate validate-all

# Individual commands
kcuda-validate detect  # GPU detection only
kcuda-validate load    # Download and load model
kcuda-validate infer "Your prompt here"  # Run inference
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
