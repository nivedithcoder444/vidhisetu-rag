"""
Unit Tests for VidhiSetu Phase 8 — OpenAI LLM Generation Client.
Verifies asynchronous generation, configuration guards, failure mode isolation,
refusal detection, and output verification with a mocked OpenAI SDK.
No live external API calls are performed.
"""

import asyncio
import os
import unittest
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import openai
from openai import AsyncOpenAI

from generation import (
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


class TestPhase8OpenAIGenerationClient(unittest.IsolatedAsyncioTestCase):
    """Test suite for OpenAIGenerationClient provider layer."""

    def setUp(self) -> None:
        self.dummy_req = httpx.Request("POST", "https://api.openai.com/v1/responses")
        self.dummy_resp = httpx.Response(500, request=self.dummy_req)
        self.valid_config = LLMConfig(
            provider="openai",
            api_key="sk-test-mock-valid-key-12345",
            model="gpt-4o-mini",
            timeout_seconds=30.0,
            max_output_tokens=1024,
        )

    def _create_mock_sdk(self, mock_response: Any = None) -> MagicMock:
        mock_sdk = MagicMock(spec=AsyncOpenAI)
        mock_sdk.responses = MagicMock()
        if mock_response is not None:
            mock_sdk.responses.create = AsyncMock(return_value=mock_response)
        else:
            mock_sdk.responses.create = AsyncMock()
        mock_sdk.close = AsyncMock()
        return mock_sdk

    # ------------------------------------------------------------------------
    # 1. Successful Generation Tests
    # ------------------------------------------------------------------------

    async def test_01_successful_generation_with_evidence(self) -> None:
        """Verify successful generation with system prompt, evidence text, and user query."""
        mock_resp = MagicMock()
        mock_resp.output_text = "Section 105 of BNS defines culpable homicide not amounting to murder."
        mock_resp.model = "gpt-4o-mini"
        mock_resp.status = "completed"
        mock_resp.refusal = None
        mock_resp.output = []
        mock_resp.id = "resp_test_001"
        mock_resp.created_at = 1700000000
        mock_resp.usage = MagicMock(input_tokens=120, output_tokens=35, total_tokens=155)

        mock_sdk = self._create_mock_sdk(mock_resp)
        client = OpenAIGenerationClient(config=self.valid_config, client=mock_sdk)

        system_instruction = "You are a legal assistant for VidhiSetu."
        user_input = "What is Section 105 of BNS?"
        evidence_text = "[BNS_Sec_105] Culpable homicide not amounting to murder..."

        result = await client.generate(
            system_instruction=system_instruction,
            user_input=user_input,
            evidence_text=evidence_text,
            max_output_tokens=512,
            timeout=15.0,
        )

        self.assertIsInstance(result, GenerationResult)
        self.assertEqual(result.text, "Section 105 of BNS defines culpable homicide not amounting to murder.")
        self.assertEqual(result.model, "gpt-4o-mini")
        self.assertEqual(result.status, "completed")
        self.assertEqual(result.prompt_tokens, 120)
        self.assertEqual(result.completion_tokens, 35)
        self.assertEqual(result.total_tokens, 155)
        self.assertEqual(result.metadata.get("id"), "resp_test_001")

        # Verify SDK call arguments
        mock_sdk.responses.create.assert_awaited_once()
        call_kwargs = mock_sdk.responses.create.await_args.kwargs
        self.assertEqual(call_kwargs["model"], "gpt-4o-mini")
        self.assertEqual(call_kwargs["instructions"], system_instruction)
        self.assertIn("Contextual Legal Evidence:", call_kwargs["input"])
        self.assertIn("User Question:", call_kwargs["input"])
        self.assertEqual(call_kwargs["max_output_tokens"], 512)
        self.assertEqual(call_kwargs["timeout"], 15.0)

    async def test_02_successful_generation_without_evidence(self) -> None:
        """Verify prompt formatting when evidence_text is omitted or None."""
        mock_resp = MagicMock()
        mock_resp.output_text = "General legal guidance response."
        mock_resp.model = "gpt-4o-mini"
        mock_resp.status = "completed"
        mock_resp.refusal = None
        mock_resp.output = []
        mock_resp.usage = None

        mock_sdk = self._create_mock_sdk(mock_resp)
        client = OpenAIGenerationClient(config=self.valid_config, client=mock_sdk)

        result = await client.generate(
            system_instruction="Assist the civilian.",
            user_input="How do I register an FIR?",
            evidence_text=None,
        )

        self.assertEqual(result.text, "General legal guidance response.")
        self.assertIsNone(result.prompt_tokens)
        call_kwargs = mock_sdk.responses.create.await_args.kwargs
        self.assertEqual(call_kwargs["input"], "How do I register an FIR?")
        self.assertNotIn("Contextual Legal Evidence:", call_kwargs["input"])

    # ------------------------------------------------------------------------
    # 2. Configuration & Credential Validation Tests
    # ------------------------------------------------------------------------

    async def test_03_missing_configuration_provider_none(self) -> None:
        """Verify that provider='none' safely raises LLMConfigurationError without making API calls."""
        cfg = LLMConfig(provider="none", api_key=None)
        client = OpenAIGenerationClient(config=cfg)

        with self.assertRaises(LLMConfigurationError) as ctx:
            await client.generate("instruction", "What is bail?")

        self.assertIn("provider is configured as 'none'", str(ctx.exception))

    async def test_04_missing_configuration_empty_api_key(self) -> None:
        """Verify that empty or whitespace API key raises LLMConfigurationError."""
        cfg_none = LLMConfig(provider="openai", api_key=None)
        client_none = OpenAIGenerationClient(config=cfg_none)
        with self.assertRaises(LLMConfigurationError) as ctx_none:
            await client_none.generate("instruction", "query")
        self.assertIn("OPENAI_API_KEY is not configured", str(ctx_none.exception))

        cfg_ws = LLMConfig(provider="openai", api_key="   ")
        client_ws = OpenAIGenerationClient(config=cfg_ws)
        with self.assertRaises(LLMConfigurationError) as ctx_ws:
            await client_ws.generate("instruction", "query")
        self.assertIn("OPENAI_API_KEY is not configured", str(ctx_ws.exception))

    def test_05_env_config_parsing_and_fallbacks(self) -> None:
        """Verify LLMConfig.from_env() parsing and sensible fallbacks."""
        with patch.dict(os.environ, {
            "LLM_PROVIDER": "openai",
            "OPENAI_API_KEY": "sk-env-test-key",
            "OPENAI_MODEL": "gpt-4o",
            "OPENAI_TIMEOUT_SECONDS": "45.5",
            "OPENAI_MAX_OUTPUT_TOKENS": "2048",
        }, clear=False):
            cfg = LLMConfig.from_env()
            self.assertEqual(cfg.provider, "openai")
            self.assertEqual(cfg.api_key, "sk-env-test-key")
            self.assertEqual(cfg.model, "gpt-4o")
            self.assertEqual(cfg.timeout_seconds, 45.5)
            self.assertEqual(cfg.max_output_tokens, 2048)

        # Fallback with invalid numerical strings
        with patch.dict(os.environ, {
            "LLM_PROVIDER": "none",
            "OPENAI_TIMEOUT_SECONDS": "invalid_float",
            "OPENAI_MAX_OUTPUT_TOKENS": "invalid_int",
        }, clear=False):
            cfg_fallback = LLMConfig.from_env()
            self.assertEqual(cfg_fallback.provider, "none")
            self.assertEqual(cfg_fallback.timeout_seconds, 30.0)
            self.assertEqual(cfg_fallback.max_output_tokens, 1024)

    # ------------------------------------------------------------------------
    # 3. Timeout Handling Tests
    # ------------------------------------------------------------------------

    async def test_06_timeout_handling_api_timeout(self) -> None:
        """Verify openai.APITimeoutError maps to LLMTimeoutError."""
        mock_sdk = self._create_mock_sdk()
        mock_sdk.responses.create = AsyncMock(
            side_effect=openai.APITimeoutError(request=self.dummy_req)
        )
        client = OpenAIGenerationClient(config=self.valid_config, client=mock_sdk)

        with self.assertRaises(LLMTimeoutError) as ctx:
            await client.generate("instruction", "query")
        self.assertIn("timed out after 30.0 seconds", str(ctx.exception))

    async def test_07_timeout_handling_asyncio_timeout(self) -> None:
        """Verify asyncio.TimeoutError maps to LLMTimeoutError."""
        mock_sdk = self._create_mock_sdk()
        mock_sdk.responses.create = AsyncMock(side_effect=asyncio.TimeoutError())
        client = OpenAIGenerationClient(config=self.valid_config, client=mock_sdk)

        with self.assertRaises(LLMTimeoutError) as ctx:
            await client.generate("instruction", "query")
        self.assertIn("timed out", str(ctx.exception))

    # ------------------------------------------------------------------------
    # 4. Rate Limiting Tests
    # ------------------------------------------------------------------------

    async def test_08_rate_limiting_error(self) -> None:
        """Verify openai.RateLimitError maps to LLMRateLimitError."""
        mock_sdk = self._create_mock_sdk()
        rate_err = openai.RateLimitError(
            message="Rate limit reached for requests per minute",
            response=httpx.Response(429, request=self.dummy_req),
            body={"error": {"message": "Rate limit exceeded"}},
        )
        mock_sdk.responses.create = AsyncMock(side_effect=rate_err)
        client = OpenAIGenerationClient(config=self.valid_config, client=mock_sdk)

        with self.assertRaises(LLMRateLimitError) as ctx:
            await client.generate("instruction", "query")
        self.assertIn("rate limit exceeded", str(ctx.exception).lower())

    # ------------------------------------------------------------------------
    # 5. Provider & Upstream Error Tests
    # ------------------------------------------------------------------------

    async def test_09_provider_connection_error(self) -> None:
        """Verify openai.APIConnectionError maps to LLMProviderError."""
        mock_sdk = self._create_mock_sdk()
        mock_sdk.responses.create = AsyncMock(
            side_effect=openai.APIConnectionError(request=self.dummy_req)
        )
        client = OpenAIGenerationClient(config=self.valid_config, client=mock_sdk)

        with self.assertRaises(LLMProviderError) as ctx:
            await client.generate("instruction", "query")
        self.assertIn("connection or server error", str(ctx.exception).lower())

    async def test_10_provider_internal_server_error(self) -> None:
        """Verify openai.InternalServerError maps to LLMProviderError."""
        mock_sdk = self._create_mock_sdk()
        server_err = openai.InternalServerError(
            message="Internal server error encountered in upstream engine",
            response=self.dummy_resp,
            body=None,
        )
        mock_sdk.responses.create = AsyncMock(side_effect=server_err)
        client = OpenAIGenerationClient(config=self.valid_config, client=mock_sdk)

        with self.assertRaises(LLMProviderError) as ctx:
            await client.generate("instruction", "query")
        self.assertIn("connection or server error", str(ctx.exception).lower())

    async def test_11_provider_generic_api_error(self) -> None:
        """Verify generic openai.APIError maps to LLMProviderError."""
        mock_sdk = self._create_mock_sdk()
        api_err = openai.APIError(
            message="Invalid parameter format",
            request=self.dummy_req,
            body=None,
        )
        mock_sdk.responses.create = AsyncMock(side_effect=api_err)
        client = OpenAIGenerationClient(config=self.valid_config, client=mock_sdk)

        with self.assertRaises(LLMProviderError) as ctx:
            await client.generate("instruction", "query")
        self.assertIn("API error: Invalid parameter format", str(ctx.exception))

    async def test_12_provider_failed_response_status(self) -> None:
        """Verify response with status='failed' raises LLMProviderError."""
        mock_resp = MagicMock()
        mock_resp.status = "failed"
        mock_resp.error = "Model generation halted by provider filter"
        mock_sdk = self._create_mock_sdk(mock_resp)
        client = OpenAIGenerationClient(config=self.valid_config, client=mock_sdk)

        with self.assertRaises(LLMProviderError) as ctx:
            await client.generate("instruction", "query")
        self.assertIn("Response status failed", str(ctx.exception))
        self.assertIn("Model generation halted by provider filter", str(ctx.exception))

    # ------------------------------------------------------------------------
    # 6. Refusal Tests
    # ------------------------------------------------------------------------

    async def test_13_refusal_attribute_on_response(self) -> None:
        """Verify response with top-level refusal attribute raises LLMRefusalError."""
        mock_resp = MagicMock()
        mock_resp.status = "completed"
        mock_resp.refusal = "I cannot provide legal instructions for unlawful activities."
        mock_resp.output = []
        mock_sdk = self._create_mock_sdk(mock_resp)
        client = OpenAIGenerationClient(config=self.valid_config, client=mock_sdk)

        with self.assertRaises(LLMRefusalError) as ctx:
            await client.generate("instruction", "query")
        self.assertIn("Model refused request", str(ctx.exception))
        self.assertIn("unlawful activities", str(ctx.exception))

    async def test_14_refusal_item_in_output_list(self) -> None:
        """Verify refusal inside response.output message items raises LLMRefusalError."""
        refusal_item = MagicMock()
        refusal_item.type = "refusal"
        refusal_item.refusal = "Content safety refusal triggered."

        mock_resp = MagicMock()
        mock_resp.status = "completed"
        mock_resp.refusal = None
        mock_resp.output = [refusal_item]
        mock_sdk = self._create_mock_sdk(mock_resp)
        client = OpenAIGenerationClient(config=self.valid_config, client=mock_sdk)

        with self.assertRaises(LLMRefusalError) as ctx:
            await client.generate("instruction", "query")
        self.assertIn("Content safety refusal triggered", str(ctx.exception))

    # ------------------------------------------------------------------------
    # 7. Empty Output Tests
    # ------------------------------------------------------------------------

    async def test_15_empty_output_string(self) -> None:
        """Verify empty output_text raises LLMEmptyOutputError."""
        mock_resp = MagicMock()
        mock_resp.status = "completed"
        mock_resp.refusal = None
        mock_resp.output = []
        mock_resp.output_text = ""
        mock_sdk = self._create_mock_sdk(mock_resp)
        client = OpenAIGenerationClient(config=self.valid_config, client=mock_sdk)

        with self.assertRaises(LLMEmptyOutputError) as ctx:
            await client.generate("instruction", "query")
        self.assertIn("empty or whitespace-only", str(ctx.exception).lower())

    async def test_16_whitespace_only_output(self) -> None:
        """Verify whitespace-only output_text raises LLMEmptyOutputError."""
        mock_resp = MagicMock()
        mock_resp.status = "completed"
        mock_resp.refusal = None
        mock_resp.output = []
        mock_resp.output_text = "   \n\t  "
        mock_sdk = self._create_mock_sdk(mock_resp)
        client = OpenAIGenerationClient(config=self.valid_config, client=mock_sdk)

        with self.assertRaises(LLMEmptyOutputError) as ctx:
            await client.generate("instruction", "query")
        self.assertIn("empty or whitespace-only", str(ctx.exception).lower())

    # ------------------------------------------------------------------------
    # 8. Incomplete Output Tests
    # ------------------------------------------------------------------------

    async def test_17_incomplete_status(self) -> None:
        """Verify response with status='incomplete' raises LLMIncompleteOutputError."""
        mock_resp = MagicMock()
        mock_resp.status = "incomplete"
        mock_resp.incomplete_details = {"reason": "max_output_tokens"}
        mock_resp.refusal = None
        mock_resp.output = []
        mock_sdk = self._create_mock_sdk(mock_resp)
        client = OpenAIGenerationClient(config=self.valid_config, client=mock_sdk)

        with self.assertRaises(LLMIncompleteOutputError) as ctx:
            await client.generate("instruction", "query")
        self.assertIn("Model response was incomplete", str(ctx.exception))
        self.assertIn("incomplete", str(ctx.exception))

    async def test_18_cancelled_status(self) -> None:
        """Verify response with status='cancelled' raises LLMIncompleteOutputError."""
        mock_resp = MagicMock()
        mock_resp.status = "cancelled"
        mock_resp.incomplete_details = None
        mock_resp.refusal = None
        mock_resp.output = []
        mock_sdk = self._create_mock_sdk(mock_resp)
        client = OpenAIGenerationClient(config=self.valid_config, client=mock_sdk)

        with self.assertRaises(LLMIncompleteOutputError) as ctx:
            await client.generate("instruction", "query")
        self.assertIn("cancelled", str(ctx.exception))

    # ------------------------------------------------------------------------
    # 9. Security, Validation & Context Manager Tests
    # ------------------------------------------------------------------------

    def test_19_security_credential_masking(self) -> None:
        """Verify that secret API keys are never leaked in LLMConfig representations."""
        secret_key = "sk-super-secret-production-token-987654321"
        cfg = LLMConfig(provider="openai", api_key=secret_key)
        rep = repr(cfg)

        self.assertNotIn(secret_key, rep)
        self.assertNotIn("987654321", rep)
        self.assertIn("api_key='***'", rep)

        cfg_none = LLMConfig(provider="none", api_key=None)
        self.assertIn("api_key='None'", repr(cfg_none))

    async def test_20_empty_user_input_validation(self) -> None:
        """Verify that empty user query raises ValueError immediately."""
        client = OpenAIGenerationClient(config=self.valid_config)
        with self.assertRaises(ValueError):
            await client.generate("instruction", "   ")
        with self.assertRaises(ValueError):
            await client.generate("instruction", "")

    async def test_21_async_context_manager_close(self) -> None:
        """Verify that the async context manager cleanly closes the client."""
        mock_resp = MagicMock()
        mock_resp.output_text = "Context manager output."
        mock_resp.model = "gpt-4o-mini"
        mock_resp.status = "completed"
        mock_resp.refusal = None
        mock_resp.output = []
        mock_resp.usage = None

        mock_sdk = self._create_mock_sdk(mock_resp)

        async with OpenAIGenerationClient(config=self.valid_config, client=mock_sdk) as client:
            res = await client.generate("instruction", "test query")
            self.assertEqual(res.text, "Context manager output.")

        mock_sdk.close.assert_awaited_once()


if __name__ == "__main__":
    unittest.main()
