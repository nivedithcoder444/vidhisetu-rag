"""
OpenAI LLM Generation Client for VidhiSetu (Phase 8).
Provides asynchronous generation using the official OpenAI Responses API.
Handles configuration, timeouts, rate limits, refusals, and output validation.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
import logging
import os
from typing import Any, Dict, Optional

import openai
from openai import AsyncOpenAI

logger = logging.getLogger("vidhisetu.generation")


# ============================================================================
# Domain Exceptions
# ============================================================================

class LLMGenerationError(Exception):
    """Base exception for all LLM generation errors."""
    pass


class LLMConfigurationError(LLMGenerationError):
    """Raised when LLM configuration or credentials are missing, disabled, or invalid."""
    pass


class LLMTimeoutError(LLMGenerationError):
    """Raised when an LLM generation request times out."""
    pass


class LLMRateLimitError(LLMGenerationError):
    """Raised when the LLM provider rate limit is exceeded."""
    pass


class LLMProviderError(LLMGenerationError):
    """Raised when the upstream LLM provider returns an API or connectivity error."""
    pass


class LLMRefusalError(LLMGenerationError):
    """Raised when the LLM model explicitly refuses to generate an answer."""
    pass


class LLMEmptyOutputError(LLMGenerationError):
    """Raised when the LLM model produces an empty or whitespace-only response."""
    pass


class LLMIncompleteOutputError(LLMGenerationError):
    """Raised when the LLM model produces an incomplete or truncated response."""
    pass


# ============================================================================
# Configuration and Result Data Classes
# ============================================================================

@dataclass
class LLMConfig:
    """Configuration for LLM generation provider."""
    provider: str = "none"
    api_key: Optional[str] = None
    model: str = "gpt-4o-mini"
    timeout_seconds: float = 30.0
    max_output_tokens: int = 1024

    @classmethod
    def from_env(cls) -> "LLMConfig":
        """Instantiate LLMConfig from environment variables."""
        provider = os.getenv("LLM_PROVIDER", "none").strip().lower()
        api_key = os.getenv("OPENAI_API_KEY")
        if api_key is not None:
            api_key = api_key.strip()
            if not api_key:
                api_key = None

        model = os.getenv("OPENAI_MODEL", "gpt-4o-mini").strip() or "gpt-4o-mini"

        timeout_raw = os.getenv("OPENAI_TIMEOUT_SECONDS", "30.0").strip()
        try:
            timeout_seconds = float(timeout_raw)
        except (ValueError, TypeError):
            timeout_seconds = 30.0

        tokens_raw = os.getenv("OPENAI_MAX_OUTPUT_TOKENS", "1024").strip()
        try:
            max_output_tokens = int(tokens_raw)
        except (ValueError, TypeError):
            max_output_tokens = 1024

        return cls(
            provider=provider,
            api_key=api_key,
            model=model,
            timeout_seconds=timeout_seconds,
            max_output_tokens=max_output_tokens,
        )

    def __repr__(self) -> str:
        """Masked string representation to avoid exposing secret API keys."""
        masked_key = "***" if self.api_key else "None"
        return (
            f"LLMConfig(provider={self.provider!r}, api_key={masked_key!r}, "
            f"model={self.model!r}, timeout_seconds={self.timeout_seconds}, "
            f"max_output_tokens={self.max_output_tokens})"
        )


@dataclass
class GenerationResult:
    """Encapsulates successful LLM generation response and execution metadata."""
    text: str
    model: str
    status: str
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    total_tokens: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


# ============================================================================
# OpenAI Client Implementation
# ============================================================================

class OpenAIGenerationClient:
    """
    Asynchronous OpenAI generation client using the official Responses API.
    Enforces secure credential management, defensive timeout handling,
    and rigorous response validation.
    """

    def __init__(
        self,
        config: Optional[LLMConfig] = None,
        client: Optional[AsyncOpenAI] = None,
    ) -> None:
        self.config = config or LLMConfig.from_env()
        self._client = client

    def _get_client(self) -> AsyncOpenAI:
        """
        Validates configuration and returns the active AsyncOpenAI instance.
        Raises LLMConfigurationError if provider is disabled or key is absent.
        """
        if self.config.provider != "openai":
            raise LLMConfigurationError(
                f"LLM provider is configured as '{self.config.provider}'. "
                f"OpenAI generation requires LLM_PROVIDER=openai."
            )

        if not self.config.api_key or not self.config.api_key.strip():
            raise LLMConfigurationError(
                "OPENAI_API_KEY is not configured or is empty. "
                "Provide a valid API key in environment or LLMConfig."
            )

        if self._client is None:
            self._client = AsyncOpenAI(api_key=self.config.api_key)

        return self._client

    async def generate(
        self,
        system_instruction: str,
        user_input: str,
        evidence_text: Optional[str] = None,
        max_output_tokens: Optional[int] = None,
        timeout: Optional[float] = None,
    ) -> GenerationResult:
        """
        Execute an asynchronous generation request via OpenAI Responses API.

        Args:
            system_instruction: System prompt controlling persona, constraints, and tone.
            user_input: Legal question or query text.
            evidence_text: Optional retrieved statutory excerpts / context.
            max_output_tokens: Optional override for token budget.
            timeout: Optional override for request timeout in seconds.

        Returns:
            GenerationResult containing the verified output text and metadata.

        Raises:
            LLMConfigurationError: Provider disabled or API key missing.
            LLMTimeoutError: Request timed out.
            LLMRateLimitError: OpenAI rate limit exceeded.
            LLMProviderError: OpenAI server/connection error or response failure.
            LLMRefusalError: Model refused the request.
            LLMIncompleteOutputError: Response generation was truncated or incomplete.
            LLMEmptyOutputError: Model returned empty or whitespace text.
        """
        if not user_input or not user_input.strip():
            raise ValueError("user_input must be a non-empty string.")

        # Configuration validation
        client = self._get_client()

        # Build bounded input context
        user_clean = user_input.strip()
        if evidence_text and evidence_text.strip():
            combined_input = (
                f"Contextual Legal Evidence:\n{evidence_text.strip()}\n\n"
                f"User Question:\n{user_clean}"
            )
        else:
            combined_input = user_clean

        instruction_clean = system_instruction.strip() if system_instruction else None
        token_limit = max_output_tokens if max_output_tokens is not None else self.config.max_output_tokens
        timeout_val = timeout if timeout is not None else self.config.timeout_seconds

        # Safe logging (metadata only - never log prompts, keys, or evidence)
        logger.info(
            "Calling OpenAI Responses API: model=%s, max_tokens=%d, "
            "timeout=%.1fs, input_len=%d, has_evidence=%s",
            self.config.model,
            token_limit,
            timeout_val,
            len(combined_input),
            bool(evidence_text and evidence_text.strip()),
        )

        try:
            call_kwargs: Dict[str, Any] = {
                "model": self.config.model,
                "input": combined_input,
                "max_output_tokens": token_limit,
                "timeout": timeout_val,
            }
            if instruction_clean:
                call_kwargs["instructions"] = instruction_clean

            response = await client.responses.create(**call_kwargs)

        except (openai.APITimeoutError, asyncio.TimeoutError) as err:
            logger.error("OpenAI generation request timed out after %.1fs", timeout_val)
            raise LLMTimeoutError(
                f"OpenAI request timed out after {timeout_val} seconds"
            ) from err

        except openai.RateLimitError as err:
            err_msg = getattr(err, "message", str(err))
            logger.error("OpenAI rate limit exceeded")
            raise LLMRateLimitError(
                f"OpenAI rate limit exceeded: {err_msg}"
            ) from err

        except (openai.APIConnectionError, openai.InternalServerError) as err:
            err_msg = getattr(err, "message", str(err))
            logger.error("OpenAI connectivity or internal server error")
            raise LLMProviderError(
                f"OpenAI connection or server error: {err_msg}"
            ) from err

        except openai.APIError as err:
            err_msg = getattr(err, "message", str(err))
            logger.error("OpenAI API error encountered")
            raise LLMProviderError(
                f"OpenAI API error: {err_msg}"
            ) from err

        except openai.OpenAIError as err:
            err_msg = getattr(err, "message", str(err))
            logger.error("OpenAI client error encountered")
            raise LLMProviderError(
                f"OpenAI client error: {err_msg}"
            ) from err

        except Exception as err:
            if isinstance(err, LLMGenerationError):
                raise
            logger.error("Unexpected error during OpenAI generation call: %s", type(err).__name__)
            raise LLMProviderError(
                f"Unexpected failure communicating with LLM provider: {str(err)}"
            ) from err

        # ====================================================================
        # Response Validation & Post-Processing
        # ====================================================================

        # 1. Response status failure check
        status = getattr(response, "status", None)
        if status == "failed":
            error_detail = getattr(response, "error", "Provider reported generation failure")
            logger.error("OpenAI response returned status 'failed'")
            raise LLMProviderError(f"Response status failed: {error_detail}")

        # 2. Refusal detection
        refusal_reason: Optional[str] = None
        if hasattr(response, "refusal") and response.refusal:
            refusal_reason = str(response.refusal)
        elif hasattr(response, "output") and response.output:
            for item in response.output:
                if getattr(item, "type", None) == "refusal" or (hasattr(item, "refusal") and item.refusal):
                    refusal_reason = getattr(item, "refusal", str(item))
                    break
                if hasattr(item, "content") and item.content:
                    for sub in item.content:
                        if getattr(sub, "type", None) == "refusal" or (hasattr(sub, "refusal") and sub.refusal):
                            refusal_reason = getattr(sub, "refusal", str(sub))
                            break
                if refusal_reason:
                    break

        if not refusal_reason and status == "refusal":
            refusal_reason = "Model refused to complete request."

        if refusal_reason:
            logger.warning("OpenAI model returned a refusal")
            raise LLMRefusalError(f"Model refused request: {refusal_reason}")

        # 3. Incomplete / Cancelled status check
        if status in ("incomplete", "cancelled"):
            details = getattr(response, "incomplete_details", None)
            logger.warning("OpenAI response incomplete: status=%s", status)
            raise LLMIncompleteOutputError(
                f"Model response was incomplete (status='{status}', details={details})"
            )

        # 4. Extract and validate output text
        output_text = getattr(response, "output_text", None)
        if output_text is None or not output_text.strip():
            logger.warning("OpenAI response contained empty or whitespace-only text")
            raise LLMEmptyOutputError("Model returned empty or whitespace-only response")

        # 5. Extract token usage metadata safely
        prompt_tokens: Optional[int] = None
        completion_tokens: Optional[int] = None
        total_tokens: Optional[int] = None
        usage = getattr(response, "usage", None)
        if usage:
            prompt_tokens = getattr(usage, "input_tokens", None)
            completion_tokens = getattr(usage, "output_tokens", None)
            total_tokens = getattr(usage, "total_tokens", None)

        metadata: Dict[str, Any] = {
            "id": getattr(response, "id", None),
            "created_at": getattr(response, "created_at", None),
        }

        logger.info(
            "OpenAI generation successful: model=%s, status=%s, "
            "tokens=(in=%s, out=%s, total=%s)",
            getattr(response, "model", self.config.model),
            status or "completed",
            prompt_tokens,
            completion_tokens,
            total_tokens,
        )

        return GenerationResult(
            text=output_text.strip(),
            model=getattr(response, "model", self.config.model),
            status=status or "completed",
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            metadata=metadata,
        )

    async def close(self) -> None:
        """Close the underlying HTTP client session if created."""
        if self._client is not None and hasattr(self._client, "close"):
            await self._client.close()

    async def __aenter__(self) -> "OpenAIGenerationClient":
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        await self.close()
