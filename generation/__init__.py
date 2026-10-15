"""
VidhiSetu Generation Package (Phase 8)
Provides LLM generation clients and configuration for statutory RAG synthesis.
"""

from generation.llm_client import (
    GenerationResult,
    LLMConfig,
    LLMConfigurationError,
    LLMEmptyOutputError,
    LLMGenerationError,
    LLMIncompleteOutputError,
    LLMProviderError,
    LLMRateLimitError,
    LLMRefusalError,
    LLMTimeoutError,
    OpenAIGenerationClient,
)

__all__ = [
    "GenerationResult",
    "LLMConfig",
    "OpenAIGenerationClient",
    "LLMGenerationError",
    "LLMConfigurationError",
    "LLMTimeoutError",
    "LLMRateLimitError",
    "LLMProviderError",
    "LLMRefusalError",
    "LLMEmptyOutputError",
    "LLMIncompleteOutputError",
]
