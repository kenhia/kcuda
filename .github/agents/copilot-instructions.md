# kcuda Development Guidelines

Auto-generated from all feature plans. Last updated: 2026-01-31

## Active Technologies

- Python 3.11+ (managed by uv) + llama-cpp-python (CUDA), torch (CUDA detection), pynvml (GPU monitoring), huggingface_hub (model downloads) (001-cuda-llm-validation)

## Project Structure

```text
src/
tests/
```

## Commands

cd src [ONLY COMMANDS FOR ACTIVE TECHNOLOGIES][ONLY COMMANDS FOR ACTIVE TECHNOLOGIES] pytest [ONLY COMMANDS FOR ACTIVE TECHNOLOGIES][ONLY COMMANDS FOR ACTIVE TECHNOLOGIES] ruff check .

## Code Style

Python 3.11+ (managed by uv): Follow standard conventions

## Recent Changes

- 001-cuda-llm-validation: Added Python 3.11+ (managed by uv) + llama-cpp-python (CUDA), torch (CUDA detection), pynvml (GPU monitoring), huggingface_hub (model downloads)

<!-- MANUAL ADDITIONS START -->

## Development Tools Policy

When a command requires a development tool that may not be installed (e.g., `jq`, `yq`, `fd`, `ripgrep`, etc.):

1. **First attempt**: Try to use the tool normally
2. **If tool is missing**: Instead of immediately working around it with alternative tools:
   - **ASK the user** if installing the tool is viable
   - Provide the install command (e.g., `sudo apt install jq`)
   - Explain why the tool would be beneficial
   - Wait for user confirmation before proceeding with workarounds
3. **Exception**: Only use workarounds immediately for universal tools (e.g., `python`, `grep`, `awk`)

**Rationale**: Installing the right tool once is better than repeatedly working around its absence. The user may prefer to have the tool available for future work.

**Example**:
```
❌ BAD: Tool 'jq' not found, using python workaround...
✅ GOOD: Tool 'jq' is not installed. Would you like me to help install it?
         It's useful for JSON parsing and would be better than workarounds.
         Install: sudo apt install jq
```

<!-- MANUAL ADDITIONS END -->
