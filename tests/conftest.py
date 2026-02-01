"""Test configuration and fixtures.

Provides mocks for optional dependencies that may not be available in CI.
"""

import sys
from unittest.mock import MagicMock

# Mock llama_cpp module for CI environments where it's not installed
# This allows @patch("llama_cpp.Llama") to work without the actual package
if "llama_cpp" not in sys.modules:
    llama_cpp_mock = MagicMock()
    llama_cpp_mock.Llama = MagicMock
    sys.modules["llama_cpp"] = llama_cpp_mock
