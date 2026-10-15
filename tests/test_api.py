import os
import sys
import unittest
from unittest.mock import AsyncMock, MagicMock
from fastapi.testclient import TestClient

# Ensure UTF-8 output on Windows console
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# Ensure project root is in sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from app.main import app
from app import main as api_main
from generation import (
    GenerationResult,
    LLMConfig,
    LLMProviderError,
    LLMConfigurationError,
)


class TestVidhiSetuAPI(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Using context manager with TestClient triggers the FastAPI lifespan event
        cls.client_ctx = TestClient(app)
        cls.client = cls.client_ctx.__enter__()

    @classmethod
    def tearDownClass(cls):
        cls.client_ctx.__exit__(None, None, None)

    def test_01_health_check(self):
        """Test GET /health returns status ok and retrieval engine status."""
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "ok")
        self.assertTrue(data["retrieval_engine_initialized"])
        self.assertGreater(data["indexed_chunks"], 4000)
        self.assertEqual(data["embedding_model"], "BAAI/bge-m3")

    def test_02_search_endpoint_valid_query(self):
        """Test POST /api/v1/search with a valid legal query."""
        payload = {"query": "breach of contract compensation", "top_k": 3}
        response = self.client.post("/api/v1/search", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["query"], "breach of contract compensation")
        self.assertTrue(len(data["passages"]) > 0)
        top = data["passages"][0]
        self.assertIn("act", top)
        self.assertIn("section_number", top)
        self.assertIn("source", top)
        self.assertIn("text", top)
        self.assertNotIn("C:\\Users", top["source"])
        self.assertNotIn("C:/Users", top["source"])

    def test_03_civilian_ask_endpoint(self):
        """Test POST /api/v1/civilian/ask returns simplified explanation and citations."""
        payload = {"query": "Section 73 of Indian Contract Act", "top_k": 2}
        response = self.client.post("/api/v1/civilian/ask", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "SUCCESS")
        self.assertTrue(len(data["simplified_explanation"]) > 0)
        self.assertTrue(len(data["key_takeaways"]) > 0)
        self.assertTrue(len(data["citations"]) > 0)
        self.assertIn("Contract Act", data["simplified_explanation"])
        for p in data["supporting_passages"]:
            self.assertNotIn("C:\\Users", p["source"])
            self.assertNotIn("C:/Users", p["source"])

    def test_04_student_ask_endpoint(self):
        """Test POST /api/v1/student/ask returns statutory analysis and principles."""
        payload = {"query": "Section 3 Arms Act", "top_k": 2}
        response = self.client.post("/api/v1/student/ask", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "SUCCESS")
        self.assertTrue(len(data["statutory_analysis"]) > 0)
        self.assertTrue(len(data["principles"]) > 0)
        self.assertIn("Arms Act", data["statutory_analysis"])

    def test_05_lawyer_research_endpoint(self):
        """Test POST /api/v1/lawyer/research returns research dossier and verbatim excerpts."""
        payload = {
            "query": "punishment for culpable homicide not amounting to murder",
            "top_k": 3,
            "act_filter": "BNS"
        }
        response = self.client.post("/api/v1/lawyer/research", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "SUCCESS")
        self.assertTrue(len(data["verbatim_excerpts"]) > 0)
        self.assertTrue(len(data["citations"]) > 0)
        top_excerpt = data["verbatim_excerpts"][0]
        self.assertIn("section", top_excerpt)
        self.assertIn("page_range", top_excerpt)
        self.assertIn("score", top_excerpt)

    def test_06_insufficient_evidence_handling(self):
        """Test that questions with absent evidence safely return INSUFFICIENT_EVIDENCE."""
        payload = {
            "query": "What is the penalty for cryptocurrency mining under BNS?",
            "top_k": 3
        }
        response = self.client.post("/api/v1/civilian/ask", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "INSUFFICIENT_EVIDENCE")
        self.assertEqual(data["supporting_passages"], [])
        self.assertEqual(data["citations"], [])

    def test_07_validation_empty_query(self):
        """Test that empty queries return HTTP 422 Unprocessable Entity."""
        res_empty = self.client.post("/api/v1/search", json={"query": ""})
        self.assertEqual(res_empty.status_code, 422)

        res_whitespace = self.client.post("/api/v1/civilian/ask", json={"query": "   "})
        self.assertEqual(res_whitespace.status_code, 422)

    def test_08_path_sanitization_across_endpoints(self):
        """Ensure no absolute host paths are exposed in any endpoint response."""
        res = self.client.post("/api/v1/search", json={"query": "injunction", "top_k": 5})
        raw_text = res.text
        self.assertNotIn("C:\\Users", raw_text)
        self.assertNotIn("C:/Users", raw_text)
        self.assertNotIn("nived", raw_text.lower())

    def test_09_grounded_generation_across_all_three_endpoints(self):
        """Verify each of the 3 endpoints maps the grounded LLM answer to its role-specific field."""
        mock_gen = MagicMock()
        mock_gen.generate = AsyncMock(return_value=GenerationResult(
            text="Grounded legal analysis synthesized for test query.",
            model="gpt-4o-mini",
            status="completed"
        ))

        original_gen = api_main.qa_pipeline_instance.generator
        try:
            api_main.qa_pipeline_instance.generator = mock_gen

            # 1. Civilian -> simplified_explanation
            res_civ = self.client.post("/api/v1/civilian/ask", json={
                "query": "Section 73 Indian Contract Act", "top_k": 2
            })
            self.assertEqual(res_civ.status_code, 200)
            data_civ = res_civ.json()
            self.assertEqual(data_civ["status"], "SUCCESS")
            self.assertIn("Grounded legal analysis synthesized for test query.", data_civ["simplified_explanation"])
            self.assertTrue(len(data_civ["key_takeaways"]) > 0)
            self.assertTrue(len(data_civ["citations"]) > 0)

            # 2. Student -> statutory_analysis
            res_stu = self.client.post("/api/v1/student/ask", json={
                "query": "Section 73 Indian Contract Act", "top_k": 2
            })
            self.assertEqual(res_stu.status_code, 200)
            data_stu = res_stu.json()
            self.assertEqual(data_stu["status"], "SUCCESS")
            self.assertIn("Grounded legal analysis synthesized for test query.", data_stu["statutory_analysis"])
            self.assertTrue(len(data_stu["principles"]) > 0)
            self.assertTrue(len(data_stu["citations"]) > 0)

            # 3. Lawyer -> research_synthesis
            res_law = self.client.post("/api/v1/lawyer/research", json={
                "query": "Section 73 Indian Contract Act", "top_k": 2
            })
            self.assertEqual(res_law.status_code, 200)
            data_law = res_law.json()
            self.assertEqual(data_law["status"], "SUCCESS")
            self.assertIn("Grounded legal analysis synthesized for test query.", data_law["research_synthesis"])
            self.assertTrue(len(data_law["verbatim_excerpts"]) > 0)
            self.assertTrue(len(data_law["citations"]) > 0)

        finally:
            api_main.qa_pipeline_instance.generator = original_gen

    def test_10_search_endpoint_never_calls_generator(self):
        """Verify POST /api/v1/search never invokes LLM generation even when generator is active."""
        mock_gen = MagicMock()
        mock_gen.generate = AsyncMock()

        original_gen = api_main.qa_pipeline_instance.generator
        try:
            api_main.qa_pipeline_instance.generator = mock_gen
            res = self.client.post("/api/v1/search", json={"query": "breach of contract", "top_k": 3})
            self.assertEqual(res.status_code, 200)
            mock_gen.generate.assert_not_called()
        finally:
            api_main.qa_pipeline_instance.generator = original_gen

    def test_11_insufficient_evidence_skips_generator_across_endpoints(self):
        """Verify queries with insufficient evidence return early without calling generator."""
        mock_gen = MagicMock()
        mock_gen.generate = AsyncMock()

        original_gen = api_main.qa_pipeline_instance.generator
        try:
            api_main.qa_pipeline_instance.generator = mock_gen
            res = self.client.post("/api/v1/civilian/ask", json={
                "query": "Interstellar mineral licensing under Indian penal law", "top_k": 3
            })
            self.assertEqual(res.status_code, 200)
            self.assertEqual(res.json()["status"], "INSUFFICIENT_EVIDENCE")
            mock_gen.generate.assert_not_called()
        finally:
            api_main.qa_pipeline_instance.generator = original_gen

    def test_12_provider_failure_maps_to_http_502(self):
        """Verify upstream LLM provider error returns controlled HTTP 502 Bad Gateway."""
        mock_gen = MagicMock()
        mock_gen.generate = AsyncMock(side_effect=LLMProviderError("Upstream OpenAI 500 error"))

        original_gen = api_main.qa_pipeline_instance.generator
        try:
            api_main.qa_pipeline_instance.generator = mock_gen
            res = self.client.post("/api/v1/civilian/ask", json={
                "query": "Section 73 Indian Contract Act", "top_k": 2
            })
            self.assertEqual(res.status_code, 502)
            self.assertIn("Upstream legal generation provider error.", res.json()["detail"])
        finally:
            api_main.qa_pipeline_instance.generator = original_gen

    def test_13_missing_configuration_failure_maps_to_http_503(self):
        """Verify missing configuration or API key returns controlled HTTP 503 Service Unavailable."""
        mock_gen = MagicMock()
        mock_gen.generate = AsyncMock(side_effect=LLMConfigurationError("OPENAI_API_KEY is not configured"))

        original_gen = api_main.qa_pipeline_instance.generator
        try:
            api_main.qa_pipeline_instance.generator = mock_gen
            res = self.client.post("/api/v1/student/ask", json={
                "query": "Section 73 Indian Contract Act", "top_k": 2
            })
            self.assertEqual(res.status_code, 503)
            self.assertIn("not properly configured", res.json()["detail"])
        finally:
            api_main.qa_pipeline_instance.generator = original_gen

    def test_14_health_check_reports_configured_provider_safely(self):
        """Verify GET /health reports provider status without making generation calls or leaking secrets."""
        mock_gen = MagicMock()
        mock_gen.config = LLMConfig(provider="openai", api_key="sk-secret-123")

        original_gen = api_main.qa_pipeline_instance.generator
        try:
            api_main.qa_pipeline_instance.generator = mock_gen
            res = self.client.get("/health")
            self.assertEqual(res.status_code, 200)
            data = res.json()
            self.assertEqual(data["llm_provider"], "openai")
            self.assertNotIn("sk-secret", res.text)
        finally:
            api_main.qa_pipeline_instance.generator = original_gen


if __name__ == "__main__":
    unittest.main()
