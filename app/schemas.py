from typing import List, Optional, Union, Dict, Any
from pydantic import BaseModel, Field, field_validator


class LegalPassage(BaseModel):
    """Structured legal chunk representation for API responses."""
    chunk_id: str = Field(..., description="Unique chunk identifier")
    act: str = Field(..., description="Name of the Legal Act")
    section_number: Union[int, str, None] = Field(None, description="Section number")
    section_title: Optional[str] = Field(None, description="Title of the section")
    chapter: Optional[str] = Field(None, description="Chapter or Part name")
    page_start: Optional[int] = Field(None, description="Starting page in source document")
    page_end: Optional[int] = Field(None, description="Ending page in source document")
    source: str = Field(..., description="Sanitized relative path or filename of source document")
    score: float = Field(..., description="Retrieval relevance score (RRF or similarity)")
    text: str = Field(..., description="Verbatim statutory text content")


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=1000, description="Legal search query")
    top_k: int = Field(default=5, ge=1, le=50, description="Maximum number of passages to return")
    act_filter: Optional[str] = Field(None, description="Optional filter restricting results to a specific Act")

    @field_validator("query")
    @classmethod
    def query_must_not_be_blank(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Query cannot be empty or whitespace only.")
        return v.strip()


class SearchResponse(BaseModel):
    query: str
    total_results: int
    passages: List[LegalPassage]


class BaseAskRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=1000, description="Legal question or lookup query")
    top_k: int = Field(default=3, ge=1, le=20, description="Number of evidence passages to retrieve")
    act_filter: Optional[str] = Field(None, description="Optional filter restricting evidence to a specific Act")

    @field_validator("query")
    @classmethod
    def query_must_not_be_blank(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Query cannot be empty or whitespace only.")
        return v.strip()


class CivilianAskResponse(BaseModel):
    status: str = Field(..., description="Status of answer generation: SUCCESS or INSUFFICIENT_EVIDENCE")
    query: str
    simplified_explanation: str
    key_takeaways: List[str]
    disclaimer: str
    supporting_passages: List[LegalPassage]
    citations: List[str]


class StudentAskResponse(BaseModel):
    status: str = Field(..., description="Status of answer generation: SUCCESS or INSUFFICIENT_EVIDENCE")
    query: str
    statutory_analysis: str
    principles: List[str]
    disclaimer: str
    supporting_passages: List[LegalPassage]
    citations: List[str]


class LawyerResearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=1000, description="Legal research query")
    top_k: int = Field(default=5, ge=1, le=50, description="Number of research passages to retrieve")
    act_filter: Optional[str] = Field(None, description="Optional filter by Act name")

    @field_validator("query")
    @classmethod
    def query_must_not_be_blank(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Query cannot be empty or whitespace only.")
        return v.strip()


class LawyerResearchResponse(BaseModel):
    status: str = Field(..., description="Status of research: SUCCESS or INSUFFICIENT_EVIDENCE")
    query: str
    research_synthesis: str
    verbatim_excerpts: List[Dict[str, Any]]
    disclaimer: str
    supporting_passages: List[LegalPassage]
    citations: List[str]


class HealthResponse(BaseModel):
    status: str
    version: str
    retrieval_engine_initialized: bool
    indexed_chunks: int
    embedding_model: str
    llm_provider: Optional[str] = Field("none", description="Active LLM provider name")
