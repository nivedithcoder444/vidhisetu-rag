import os
import sys
import json
import unittest

# Ensure UTF-8 output on Windows console
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# Ensure project root is in sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from retrieval.hybrid import HybridRetriever
from retrieval.answer_pipeline import SourceGroundedQAPipeline


class TestPhase7LegalRetrievalQuality(unittest.TestCase):
    """
    Phase 7 Regression and Retrieval Quality Test Suite for VidhiSetu.
    Validates section parser accuracy, table-of-contents isolation,
    precise statutory page ranges, act filtering, citation integrity,
    and anti-hallucination guardrails for missing evidence.
    """

    @classmethod
    def setUpClass(cls):
        cls.retriever = HybridRetriever(
            chunks_path="data/processed/all_legal_chunks.json",
            embeddings_path="data/processed/all_legal_embeddings.npy",
            model_name="BAAI/bge-m3"
        )
        cls.retriever.initialize()
        cls.qa_pipeline = SourceGroundedQAPipeline(hybrid_retriever=cls.retriever)

        # Also load chunks directly for structural validation
        chunks_file = os.path.join(project_root, "data/processed/all_legal_chunks.json")
        with open(chunks_file, 'r', encoding='utf-8') as f:
            cls.all_chunks = json.load(f)

    def test_01_no_excessively_broad_page_span_chunks(self):
        """Verify no chunk spans an excessively broad page range (>8 pages) or swallows the Act."""
        broad_chunks = [
            c for c in self.all_chunks
            if (c.get("page_end", 0) - c.get("page_start", 0)) > 8
        ]
        self.assertEqual(
            len(broad_chunks), 0,
            f"Found {len(broad_chunks)} chunks with page span > 8 pages: {[c['chunk_id'] for c in broad_chunks[:5]]}"
        )

        # Check BNS Section 358 specifically (previously swallowed pages 15-112)
        bns_358_chunks = [
            c for c in self.all_chunks
            if "BNS" in c.get("document_id", "") and c.get("section_number") == 358
        ]
        for c in bns_358_chunks:
            span = c.get("page_end", 0) - c.get("page_start", 0)
            self.assertLessEqual(span, 3, f"BNS Section 358 chunk {c['chunk_id']} has excessive span {span}")

    def test_02_bns_section_105_exact_retrieval(self):
        """
        Verify BNS Section 105 retrieves statutory text from body (Page 47),
        NOT a 1-line Table of Contents entry from Page 6 or Section 358.
        """
        results = self.retriever.search("Section 105 BNS", top_k=3)
        self.assertTrue(len(results) > 0)
        top = results[0]

        self.assertEqual(top.get("section_number"), 105)
        self.assertTrue(
            "nyaya" in top.get("act", "").lower() or "bns" in top.get("document_id", "").lower()
        )
        # Should be on page 47 in the official BNS 2023 Gazette PDF
        self.assertEqual(top.get("page_start"), 47)
        self.assertEqual(top.get("page_end"), 47)

        text_lower = top.get("text", "").lower()
        self.assertIn("culpable homicide", text_lower)
        self.assertIn("not amounting to murder", text_lower)

        # Ensure it is NOT the Table of Contents 1-liner
        self.assertGreater(len(top.get("text", "")), 150)

    def test_03_contract_act_section_73(self):
        """
        Verify Indian Contract Act Section 73 retrieves substantive body text (Pages 29-31),
        NOT Table of Contents from Page 4.
        """
        results = self.retriever.search("Section 73 of Indian Contract Act", top_k=3)
        self.assertTrue(len(results) > 0)
        top = results[0]

        self.assertEqual(top.get("section_number"), 73)
        self.assertIn("contract", top.get("act", "").lower())
        self.assertGreaterEqual(top.get("page_start"), 28)
        self.assertLessEqual(top.get("page_end"), 32)

        text_lower = top.get("text", "").lower()
        self.assertIn("compensation", text_lower)
        self.assertIn("breach", text_lower)

    def test_04_arms_act_section_3(self):
        """
        Verify Arms Act Section 3 retrieves substantive body text (Pages 3-4),
        NOT Table of Contents from Page 1.
        """
        results = self.retriever.search("Section 3 Arms Act", top_k=3)
        self.assertTrue(len(results) > 0)
        top = results[0]

        self.assertEqual(top.get("section_number"), 3)
        self.assertIn("arms", top.get("act", "").lower())
        self.assertIn(top.get("page_start"), [3, 4])

        text_lower = top.get("text", "").lower()
        self.assertTrue("firearms" in text_lower or "ammunition" in text_lower or "licence" in text_lower)

    def test_05_act_filter_consistency(self):
        """Verify act_filter strictly restricts candidates across all retrieval layers."""
        bns_results = self.retriever.search(
            "punishment for murder",
            top_k=5,
            act_filter="Bharatiya Nyaya Sanhita"
        )
        self.assertTrue(len(bns_results) > 0)
        for r in bns_results:
            self.assertTrue(
                "nyaya" in r.get("act", "").lower() or "bns" in r.get("document_id", "").lower(),
                f"Result {r['chunk_id']} from {r.get('act')} violated act_filter"
            )

    def test_06_insufficient_evidence_safeguard(self):
        """Verify queries with no local statutory basis return INSUFFICIENT_EVIDENCE."""
        absent_queries = [
            "What is the statutory penalty for interstellar asteroid mining under BNS?",
            "quantum mechanical wave function optimization for deep space communication"
        ]
        for q in absent_queries:
            qa_res = self.qa_pipeline.answer_question(q, top_k=3)
            self.assertEqual(
                qa_res["status"], "INSUFFICIENT_EVIDENCE",
                f"Query '{q}' should have returned INSUFFICIENT_EVIDENCE but got {qa_res['status']}"
            )
            self.assertEqual(qa_res["supporting_chunks"], [])
            self.assertEqual(qa_res["citations"], [])

    def test_07_citation_integrity_and_alignment(self):
        """
        Verify every citation directly aligns with the supporting passage
        and includes Act name, section number, title, and exact page information.
        """
        qa_res = self.qa_pipeline.answer_question("Section 105 BNS", top_k=2)
        self.assertEqual(qa_res["status"], "SUCCESS")
        self.assertTrue(len(qa_res["citations"]) > 0)

        top_citation = qa_res["citations"][0]
        top_chunk = qa_res["supporting_chunks"][0]

        # Verify citation matches chunk metadata
        self.assertIn("Section 105", top_citation)
        self.assertIn("Page 47", top_citation)
        self.assertEqual(top_chunk["section_number"], 105)
        self.assertEqual(top_chunk["page_start"], 47)

        # Verify answer text cites the top fact
        self.assertIn("Section 105", qa_res["answer"])


if __name__ == "__main__":
    unittest.main()
