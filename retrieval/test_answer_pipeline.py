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

from retrieval.answer_pipeline import SourceGroundedQAPipeline


class TestSourceGroundedQAPipeline(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.qa_pipeline = SourceGroundedQAPipeline()

    def test_valid_question_answering(self):
        res = self.qa_pipeline.answer_question("What is the penalty for breach of contract?")
        self.assertEqual(res["status"], "SUCCESS")
        self.assertTrue(len(res["facts"]) > 0)
        self.assertTrue(len(res["citations"]) > 0)
        self.assertTrue(len(res["supporting_chunks"]) > 0)
        self.assertIn("Section", res["answer"])

    def test_empty_query_handling(self):
        res = self.qa_pipeline.answer_question("")
        self.assertEqual(res["status"], "INVALID_QUERY")

    def test_absent_evidence_handling(self):
        res = self.qa_pipeline.answer_question("What is the penalty for cryptocurrency mining under BNS?")
        self.assertEqual(res["status"], "INSUFFICIENT_EVIDENCE")
        self.assertEqual(res["facts"], [])
        self.assertEqual(res["supporting_chunks"], [])

    def test_citation_metadata_integrity(self):
        res = self.qa_pipeline.answer_question("Licence for acquisition of firearms under Arms Act")
        self.assertEqual(res["status"], "SUCCESS")
        top_chunk = res["supporting_chunks"][0]
        self.assertIn("act", top_chunk)
        self.assertIn("section_number", top_chunk)
        self.assertIn("source", top_chunk)
        self.assertIn("page_start", top_chunk)
        self.assertIn("page_end", top_chunk)


if __name__ == "__main__":
    unittest.main()
