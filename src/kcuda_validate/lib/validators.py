"""Input validation utilities."""


def validate_prompt(prompt: str) -> tuple[bool, str | None]:
    """
    Validate user prompt for inference.

    Args:
        prompt: User input text

    Returns:
        Tuple of (is_valid, error_message)
    """
    if not prompt or not prompt.strip():
        return False, "Prompt cannot be empty"

    if len(prompt) > 10000:
        return False, f"Prompt too long: {len(prompt)} characters (max 10000)"

    return True, None


def validate_repo_id(repo_id: str) -> tuple[bool, str | None]:
    """
    Validate Hugging Face repository ID format.

    Args:
        repo_id: Repository identifier (e.g., "username/model-name")

    Returns:
        Tuple of (is_valid, error_message)
    """
    if not repo_id:
        return False, "Repository ID cannot be empty"

    # Basic format: username/repo-name
    if "/" not in repo_id:
        return False, f"Invalid repo ID format: {repo_id} (expected 'username/repo-name')"

    parts = repo_id.split("/")
    if len(parts) != 2:
        return False, f"Invalid repo ID format: {repo_id} (expected 'username/repo-name')"

    username, repo_name = parts
    if not username or not repo_name:
        return False, f"Invalid repo ID format: {repo_id} (username or repo-name empty)"

    return True, None


def validate_filename(filename: str) -> tuple[bool, str | None]:
    """
    Validate GGUF filename format.

    Args:
        filename: Model filename

    Returns:
        Tuple of (is_valid, error_message)
    """
    if not filename:
        return False, "Filename cannot be empty"

    if not filename.endswith(".gguf"):
        return False, f"Invalid file extension: {filename} (expected .gguf)"

    # Check for invalid path characters
    if any(c in filename for c in ["/", "\\", ".."]):
        return False, f"Invalid filename: {filename} (contains path separators)"

    return True, None


def validate_gpu_id(device_id: int) -> tuple[bool, str | None]:
    """
    Validate GPU device ID.

    Args:
        device_id: GPU device index

    Returns:
        Tuple of (is_valid, error_message)
    """
    if device_id < 0:
        return False, f"Invalid GPU device ID: {device_id} (must be >= 0)"

    if device_id > 15:  # Reasonable upper limit
        return False, f"Invalid GPU device ID: {device_id} (must be <= 15)"

    return True, None


def validate_temperature(temperature: float) -> tuple[bool, str | None]:
    """
    Validate sampling temperature.

    Args:
        temperature: Sampling temperature value

    Returns:
        Tuple of (is_valid, error_message)
    """
    if temperature < 0.0:
        return False, f"Invalid temperature: {temperature} (must be >= 0.0)"

    if temperature > 2.0:
        return False, f"Invalid temperature: {temperature} (must be <= 2.0)"

    return True, None


def validate_max_tokens(max_tokens: int) -> tuple[bool, str | None]:
    """
    Validate max tokens parameter.

    Args:
        max_tokens: Maximum tokens to generate

    Returns:
        Tuple of (is_valid, error_message)
    """
    if max_tokens <= 0:
        return False, f"Invalid max_tokens: {max_tokens} (must be > 0)"

    if max_tokens > 4096:  # Reasonable upper limit
        return False, f"Invalid max_tokens: {max_tokens} (must be <= 4096)"

    return True, None
