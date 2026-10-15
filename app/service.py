import os
import re
from typing import List, Dict, Any, Optional

from fastapi import HTTPException, status

from app.schemas import (
    LegalPassage,
    CivilianAskResponse,
    StudentAskResponse,
    LawyerResearchResponse,
    SearchResponse
)
from retrieval.hybrid import HybridRetriever
from retrieval.answer_pipeline import SourceGroundedQAPipeline


def sanitize_source_path(source: str) -> str:
    """
    Sanitizes source file path to prevent leaking local filesystem structure.
    Returns clean relative path e.g. 'data/raw/extracted/indian contract act 1872.pdf'.
    """
    if not source:
        return ""
    norm = source.replace("\\", "/")
    if "data/raw/extracted/" in norm:
        return "data/raw/extracted/" + norm.split("data/raw/extracted/")[-1]
    elif "data/" in norm:
        return "data/" + norm.split("data/")[-1]
    return os.path.basename(norm)


def sanitize_citation_string(citation: str) -> str:
    """Sanitizes full citation string preserving Act name and section while cleaning path."""
    if not citation:
        return ""
    norm = citation.replace("\\", "/")
    return re.sub(r'[A-Za-z]:/[^\]\[]+/data/', 'data/', norm)


def format_legal_passage(chunk: Dict[str, Any]) -> LegalPassage:
    """Formats a raw retrieval chunk into a validated LegalPassage schema."""
    return LegalPassage(
        chunk_id=chunk.get("chunk_id", ""),
        act=chunk.get("act", "Unspecified Act"),
        section_number=chunk.get("section_number"),
        section_title=chunk.get("section_title"),
        chapter=chunk.get("chapter"),
        page_start=chunk.get("page_start"),
        page_end=chunk.get("page_end"),
        source=sanitize_source_path(chunk.get("source", "")),
        score=float(chunk.get("rrf_score", chunk.get("score", 0.0))),
        text=chunk.get("text", "").strip()
    )


def handle_pipeline_status(qa_result: Dict[str, Any]) -> None:
    """
    Evaluates QA pipeline status and maps errors to controlled HTTP exceptions.
    Prevents leaking internal stack traces, API keys, or raw system prompts.
    """
    status_val = qa_result.get("status", "")
    if status_val == "GENERATION_ERROR":
        error_msg = qa_result.get("error", "")
        if "not configured" in error_msg.lower() or "configuration" in error_msg.lower():
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Legal answer generation service is not properly configured."
            )
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Upstream legal generation provider error."
        )
    if status_val not in ("SUCCESS", "FALLBACK_DETERMINISTIC", "INSUFFICIENT_EVIDENCE"):
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to synthesize legal answer."
        )


class LegalAPIService:
    """
    Service layer providing role-specific legal assistance policies
    using the shared hybrid retrieval and source-grounded QA pipeline.
    """

    def __init__(self, retriever: HybridRetriever, qa_pipeline: SourceGroundedQAPipeline):
        self.retriever = retriever
        self.qa_pipeline = qa_pipeline

    def search_passages(self, query: str, top_k: int = 5, act_filter: Optional[str] = None) -> SearchResponse:
        """Raw hybrid search endpoint returning ranked legal passages."""
        raw_results = self.retriever.search(query, top_k=top_k, act_filter=act_filter)
        passages = [format_legal_passage(r) for r in raw_results]
        return SearchResponse(
            query=query,
            total_results=len(passages),
            passages=passages
        )

    def process_civilian_query(self, query: str, top_k: int = 3, act_filter: Optional[str] = None) -> CivilianAskResponse:
        """
        Civilian Policy:
        - Plain language framing using grounded answer
        - Actionable statutory takeaway bullets derived safely from facts
        - Legal disclaimer
        - Full citation grounding without hallucinating LLM answers
        """
        qa_result = self.qa_pipeline.answer_question(query, top_k=top_k, act_filter=act_filter)
        handle_pipeline_status(qa_result)

        if qa_result.get("status") == "INSUFFICIENT_EVIDENCE":
            return CivilianAskResponse(
                status="INSUFFICIENT_EVIDENCE",
                query=query,
                simplified_explanation="Insufficient authoritative legal evidence was found in the ingested VidhiSetu dataset to address this query safely without risk of inaccuracy.",
                key_takeaways=[],
                disclaimer="VidhiSetu does not synthesize unverified legal claims when source evidence is missing.",
                supporting_passages=[],
                citations=[]
            )

        supporting_chunks = qa_result.get("supporting_chunks", [])
        facts = qa_result.get("facts", [])
        top_fact = facts[0] if facts else None

        passages = [format_legal_passage(chunk) for chunk in supporting_chunks]

        # Role mapping: Use grounded LLM synthesis if available, else deterministic framing
        if qa_result.get("generation_mode") == "llm_grounded":
            simplified = qa_result.get("answer", "")
        elif top_fact:
            sec_label = f"Section {top_fact['section']}" if top_fact.get('section') else "relevant provision"
            simplified = (
                f"Under the {top_fact['act']}, {sec_label} ('{top_fact['section_title']}'), "
                f"the law provides as follows: {top_fact['verbatim_text']}"
            )
        else:
            simplified = qa_result.get("answer", "")

        # Key takeaways: safe deterministic derivation from top verified fact
        if top_fact:
            sec_label = f"Section {top_fact['section']}" if top_fact.get('section') else "relevant provision"
            takeaways = [
                f"Governing Authority: {top_fact['act']}, {sec_label}",
                f"Official Subject: {top_fact['section_title'] or 'General Provision'}",
                f"Source Citation: {top_fact['citation']}"
            ]
        else:
            takeaways = []

        clean_citations = [
            sanitize_citation_string(c)
            for c in qa_result.get("citations", [])
        ]

        return CivilianAskResponse(
            status="SUCCESS",
            query=query,
            simplified_explanation=simplified,
            key_takeaways=takeaways,
            disclaimer="Notice: This summary is synthesized strictly from local authoritative legal documents for public educational purposes and does not constitute formal legal counsel.",
            supporting_passages=passages,
            citations=clean_citations
        )

    def process_student_query(self, query: str, top_k: int = 3, act_filter: Optional[str] = None) -> StudentAskResponse:
        """
        Student Policy:
        - Statutory context & chapter hierarchy using grounded answer
        - Essential legal principles & statutory elements derived safely from evidence
        - Pedagogical reference framing
        """
        qa_result = self.qa_pipeline.answer_question(query, top_k=top_k, act_filter=act_filter)
        handle_pipeline_status(qa_result)

        if qa_result.get("status") == "INSUFFICIENT_EVIDENCE":
            return StudentAskResponse(
                status="INSUFFICIENT_EVIDENCE",
                query=query,
                statutory_analysis="Insufficient authoritative legal material found in the ingested corpus for academic analysis of this topic.",
                principles=[],
                disclaimer="VidhiSetu requires verified legislative sources to prevent conceptual errors in legal study.",
                supporting_passages=[],
                citations=[]
            )

        supporting_chunks = qa_result.get("supporting_chunks", [])
        passages = [format_legal_passage(chunk) for chunk in supporting_chunks]
        top_passage = passages[0] if passages else None

        # Role mapping: Use grounded LLM synthesis if available, else deterministic framing
        if qa_result.get("generation_mode") == "llm_grounded":
            statutory_analysis = qa_result.get("answer", "")
        elif top_passage:
            chapter_info = f" in {top_passage.chapter}" if top_passage.chapter else ""
            statutory_analysis = (
                f"Statutory Analysis of {top_passage.act}{chapter_info}, Section {top_passage.section_number} "
                f"('{top_passage.section_title}'):\n\n"
                f"\"{top_passage.text}\""
            )
        else:
            statutory_analysis = qa_result.get("answer", "")

        # Principles: safe deterministic derivation from top passage
        if top_passage:
            principles = [
                f"Statutory Scheme: {top_passage.act} ({top_passage.chapter or 'General'})",
                f"Legislative Provision: Section {top_passage.section_number} - {top_passage.section_title}",
                f"Textual Authority: Verified from {top_passage.source} (Page {top_passage.page_start}-{top_passage.page_end})"
            ]
        else:
            principles = []

        clean_citations = [
            sanitize_citation_string(c)
            for c in qa_result.get("citations", [])
        ]

        return StudentAskResponse(
            status="SUCCESS",
            query=query,
            statutory_analysis=statutory_analysis,
            principles=principles,
            disclaimer="Academic Reference: Sourced strictly from verified statutory enactments for legal study and research analysis.",
            supporting_passages=passages,
            citations=clean_citations
        )

    def process_lawyer_query(
        self,
        query: str,
        top_k: int = 5,
        act_filter: Optional[str] = None
    ) -> LawyerResearchResponse:
        """
        Lawyer Policy:
        - Strict research dossier using grounded answer for research_synthesis
        - Full verbatim excerpts with scores & page ranges
        - Consistent Act-level filtering across all candidates
        - No speculative commentary
        """
        qa_result = self.qa_pipeline.answer_question(query, top_k=top_k, act_filter=act_filter)
        handle_pipeline_status(qa_result)

        if qa_result.get("status") == "INSUFFICIENT_EVIDENCE":
            return LawyerResearchResponse(
                status="INSUFFICIENT_EVIDENCE",
                query=query,
                research_synthesis="No statutory authority or judicial enactment located in the local indexed dataset satisfying the query criteria.",
                verbatim_excerpts=[],
                disclaimer="Strict grounding policy: VidhiSetu will not fabricate statutory provisions, judgments, or citations.",
                supporting_passages=[],
                citations=[]
            )

        supporting_chunks = qa_result.get("supporting_chunks", [])
        passages = [format_legal_passage(chunk) for chunk in supporting_chunks]
        verbatim_excerpts = []
        clean_citations = [
            sanitize_citation_string(c)
            for c in qa_result.get("citations", [])
        ]

        for p in passages:
            page_info = f"Pages {p.page_start}-{p.page_end}" if p.page_start != p.page_end else f"Page {p.page_start}"
            verbatim_excerpts.append({
                "act": p.act,
                "section": p.section_number,
                "section_title": p.section_title,
                "source": p.source,
                "page_range": page_info,
                "score": p.score,
                "text": p.text
            })

        # Role mapping: Use grounded LLM synthesis if available, else deterministic synthesis
        if qa_result.get("generation_mode") == "llm_grounded":
            synthesis = qa_result.get("answer", "")
        else:
            synthesis = (
                f"Statutory Research Dossier for query: '{query}'. "
                f"Identified {len(passages)} authoritative statutory provisions across "
                f"{', '.join(set(p.act for p in passages))}."
            )

        return LawyerResearchResponse(
            status="SUCCESS",
            query=query,
            research_synthesis=synthesis,
            verbatim_excerpts=verbatim_excerpts,
            disclaimer="Professional Legal Research Notice: Transcribed verbatim from local indexed statutory enactments. Citing counsel must independently verify current Gazette status.",
            supporting_passages=passages,
            citations=clean_citations
        )
