import unittest
import os
import sys

# Ensure UTF-8 output on Windows console
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# Ensure project root is in sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from retrieval.hybrid import HybridRetriever, parse_exact_section_query


class TestHybridRetriever(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.chunks_path = "data/processed/all_legal_chunks.json"
        cls.embeddings_path = "data/processed/all_legal_embeddings.npy"
        cls.retriever = HybridRetriever(
            chunks_path=cls.chunks_path,
            embeddings_path=cls.embeddings_path
        )
        cls.retriever.initialize()

    def test_parse_exact_section_query(self):
        sec, act = parse_exact_section_query("Section 73 of Indian Contract Act")
        self.assertEqual(sec, 73)
        self.assertIn("contract", act.lower())

        sec_arms, act_arms = parse_exact_section_query("Section 3 Arms Act")
        self.assertEqual(sec_arms, 3)
        self.assertIn("arms", act_arms.lower())

        sec_none, act_none = parse_exact_section_query("What is breach of contract?")
        self.assertIsNone(sec_none)

    def test_empty_query_handling(self):
        res_empty = self.retriever.search("")
        self.assertEqual(res_empty, [])

        res_spaces = self.retriever.search("   ")
        self.assertEqual(res_spaces, [])

    def test_deduplication(self):
        results = self.retriever.search("breach of contract compensation", top_k=10)
        chunk_ids = [r["chunk_id"] for r in results]
        self.assertEqual(len(chunk_ids), len(set(chunk_ids)), "Duplicate chunk IDs found in hybrid search results!")

    def test_exact_section_retrieval(self):
        results = self.retriever.search("Section 73 Indian Contract Act", top_k=3)
        self.assertTrue(len(results) > 0)
        top_result = results[0]
        self.assertEqual(top_result["section_number"], 73)
        self.assertIn("Contract Act", top_result["act"])

    def test_rrf_scoring(self):
        results = self.retriever.search("licence for acquisition of firearms", top_k=5)
        self.assertTrue(len(results) > 0)
        for i in range(len(results) - 1):
            self.assertGreaterEqual(results[i]["rrf_score"], results[i + 1]["rrf_score"])

    def test_metadata_preservation(self):
        results = self.retriever.search("punishment for murder", top_k=3)
        self.assertTrue(len(results) > 0)
        top = results[0]
        self.assertIn("act", top)
        self.assertIn("source", top)
        self.assertIn("section_number", top)
        self.assertIn("section_title", top)
        self.assertIn("page_start", top)
        self.assertIn("page_end", top)
        self.assertIn("text", top)


if __name__ == "__main__":
    unittest.main()
