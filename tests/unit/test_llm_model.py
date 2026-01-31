"""Unit tests for LLMModel data model.

Tests validation rules and constraints for the LLM model entity
as defined in data-model.md.
"""

import pytest
from kcuda_validate.models.llm_model import LLMModel


class TestLLMModelValidation:
    """Test LLMModel validation rules per data-model.md."""

    def test_valid_llm_model_creation(self):
        """Test creating a valid LLMModel with all required attributes."""
        model = LLMModel(
            repo_id="Ttimofeyka/MistralRP-Noromaid-NSFW-Mistral-7B-GGUF",
            filename="mistralrp-noromaid-nsfw-mistral-7b.Q4_K_M.gguf",
            local_path="/home/user/.cache/huggingface/hub/models/model.gguf",
            file_size_mb=4168,
            parameter_count=7_240_000_000,
            quantization_type="Q4_K_M",
            context_length=8192,
            vram_usage_mb=0,  # Not yet loaded
            is_loaded=False,
        )

        assert model.repo_id == "Ttimofeyka/MistralRP-Noromaid-NSFW-Mistral-7B-GGUF"
        assert model.filename == "mistralrp-noromaid-nsfw-mistral-7b.Q4_K_M.gguf"
        assert model.file_size_mb == 4168
        assert model.parameter_count == 7_240_000_000
        assert model.quantization_type == "Q4_K_M"
        assert model.context_length == 8192
        assert model.vram_usage_mb == 0
        assert not model.is_loaded

    def test_file_size_must_be_positive(self):
        """Test that file_size_mb must be > 0."""
        with pytest.raises(ValueError, match=r"[Ff]ile size.*positive"):
            LLMModel(
                repo_id="test/repo",
                filename="model.gguf",
                local_path="/path/to/model.gguf",
                file_size_mb=0,  # Invalid - must be > 0
                parameter_count=7_000_000_000,
                quantization_type="Q4_K_M",
                context_length=8192,
                vram_usage_mb=0,
                is_loaded=False,
            )

    def test_file_size_cannot_be_negative(self):
        """Test that file_size_mb cannot be negative."""
        with pytest.raises(ValueError, match=r"[Ff]ile size.*positive"):
            LLMModel(
                repo_id="test/repo",
                filename="model.gguf",
                local_path="/path/to/model.gguf",
                file_size_mb=-100,  # Invalid - negative
                parameter_count=7_000_000_000,
                quantization_type="Q4_K_M",
                context_length=8192,
                vram_usage_mb=0,
                is_loaded=False,
            )

    def test_context_length_must_be_positive(self):
        """Test that context_length must be > 0."""
        with pytest.raises(ValueError, match=r"[Cc]ontext length.*positive"):
            LLMModel(
                repo_id="test/repo",
                filename="model.gguf",
                local_path="/path/to/model.gguf",
                file_size_mb=4168,
                parameter_count=7_000_000_000,
                quantization_type="Q4_K_M",
                context_length=0,  # Invalid - must be > 0
                vram_usage_mb=0,
                is_loaded=False,
            )

    def test_vram_usage_valid_when_loaded(self):
        """Test that vram_usage_mb must be > 0 when is_loaded=True."""
        with pytest.raises(
            ValueError, match=r"[Vv][Rr][Aa][Mm].*positive.*loaded|loaded.*[Vv][Rr][Aa][Mm].*positive"
        ):
            LLMModel(
                repo_id="test/repo",
                filename="model.gguf",
                local_path="/path/to/model.gguf",
                file_size_mb=4168,
                parameter_count=7_000_000_000,
                quantization_type="Q4_K_M",
                context_length=8192,
                vram_usage_mb=0,  # Invalid when is_loaded=True
                is_loaded=True,
            )

    def test_quantization_type_valid_formats(self):
        """Test that quantization_type accepts valid GGUF formats."""
        valid_quants = ["Q4_K_M", "Q4_K_S", "Q5_K_M", "Q5_K_S", "Q8_0", "F16", "F32"]

        for quant in valid_quants:
            model = LLMModel(
                repo_id="test/repo",
                filename=f"model.{quant}.gguf",
                local_path="/path/to/model.gguf",
                file_size_mb=4168,
                parameter_count=7_000_000_000,
                quantization_type=quant,
                context_length=8192,
                vram_usage_mb=0,
                is_loaded=False,
            )
            assert model.quantization_type == quant

    def test_quantization_type_cannot_be_empty(self):
        """Test that quantization_type must not be empty."""
        with pytest.raises(ValueError, match=r"[Qq]uantization.*empty"):
            LLMModel(
                repo_id="test/repo",
                filename="model.gguf",
                local_path="/path/to/model.gguf",
                file_size_mb=4168,
                parameter_count=7_000_000_000,
                quantization_type="",  # Invalid - empty
                context_length=8192,
                vram_usage_mb=0,
                is_loaded=False,
            )

    def test_repo_id_cannot_be_empty(self):
        """Test that repo_id must not be empty."""
        with pytest.raises(ValueError, match=r"[Rr]epo.*[Ii][Dd].*empty"):
            LLMModel(
                repo_id="",  # Invalid - empty
                filename="model.gguf",
                local_path="/path/to/model.gguf",
                file_size_mb=4168,
                parameter_count=7_000_000_000,
                quantization_type="Q4_K_M",
                context_length=8192,
                vram_usage_mb=0,
                is_loaded=False,
            )

    def test_filename_cannot_be_empty(self):
        """Test that filename must not be empty."""
        with pytest.raises(ValueError, match=r"[Ff]ilename.*empty"):
            LLMModel(
                repo_id="test/repo",
                filename="",  # Invalid - empty
                local_path="/path/to/model.gguf",
                file_size_mb=4168,
                parameter_count=7_000_000_000,
                quantization_type="Q4_K_M",
                context_length=8192,
                vram_usage_mb=0,
                is_loaded=False,
            )

    def test_llm_model_immutability(self):
        """Test that LLMModel is immutable after creation (per data-model.md)."""
        model = LLMModel(
            repo_id="test/repo",
            filename="model.gguf",
            local_path="/path/to/model.gguf",
            file_size_mb=4168,
            parameter_count=7_000_000_000,
            quantization_type="Q4_K_M",
            context_length=8192,
            vram_usage_mb=0,
            is_loaded=False,
        )

        # Should not be able to modify attributes
        with pytest.raises(AttributeError):
            model.is_loaded = True

    def test_llm_model_repr(self):
        """Test that LLMModel has readable repr."""
        model = LLMModel(
            repo_id="test/repo",
            filename="model.Q4_K_M.gguf",
            local_path="/path/to/model.gguf",
            file_size_mb=4168,
            parameter_count=7_240_000_000,
            quantization_type="Q4_K_M",
            context_length=8192,
            vram_usage_mb=4832,
            is_loaded=True,
        )

        repr_str = repr(model)
        assert "LLMModel" in repr_str
        assert "test/repo" in repr_str
        assert "Q4_K_M" in repr_str
