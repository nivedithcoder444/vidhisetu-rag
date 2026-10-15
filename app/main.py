import os
import sys
import logging
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

# Ensure project root is in sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from retrieval.hybrid import HybridRetriever
from retrieval.answer_pipeline import SourceGroundedQAPipeline
from app.schemas import (
    HealthResponse,
    SearchRequest,
    SearchResponse,
    BaseAskRequest,
    CivilianAskResponse,
    StudentAskResponse,
    LawyerResearchRequest,
    LawyerResearchResponse
)
from app.service import LegalAPIService

# Configure safe structured logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("vidhisetu.api")


# Singleton instances managed via lifespan
retriever_instance: Optional[HybridRetriever] = None
qa_pipeline_instance: Optional[SourceGroundedQAPipeline] = None
legal_service_instance: Optional[LegalAPIService] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager that initializes the retrieval engine
    and embeddings once at startup, avoiding re-indexing per request.
    """
    global retriever_instance, qa_pipeline_instance, legal_service_instance
    logger.info("Initializing VidhiSetu Hybrid Retrieval Engine...")
    
    try:
        retriever_instance = HybridRetriever(
            chunks_path="data/processed/all_legal_chunks.json",
            embeddings_path="data/processed/all_legal_embeddings.npy",
            model_name="BAAI/bge-m3"
        )
        retriever_instance.initialize()
        
        qa_pipeline_instance = SourceGroundedQAPipeline(hybrid_retriever=retriever_instance)
        legal_service_instance = LegalAPIService(
            retriever=retriever_instance,
            qa_pipeline=qa_pipeline_instance
        )
        
        chunk_count = len(retriever_instance.bm25_retriever.chunks)
        logger.info(f"VidhiSetu Retrieval Engine successfully initialized with {chunk_count} chunks.")
    except Exception as e:
        logger.error(f"Failed to initialize retrieval engine during startup: {e}", exc_info=True)
        raise

    yield

    logger.info("Shutting down VidhiSetu API service...")


app = FastAPI(
    title="VidhiSetu Legal AI RAG API",
    description="Authoritative, source-grounded Indian legal retrieval and assistance API.",
    version="1.0.0",
    lifespan=lifespan
)

# Enable CORS for local and standard web clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """
    Global exception handler that prevents leaking stack traces,
    passwords, internal paths, or API keys in public responses.
    """
    logger.error(f"Unhandled exception while processing {request.method} {request.url.path}: {exc}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "An internal server error occurred while processing the legal query."}
    )


@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """System health check and retrieval engine status."""
    is_ready = (
        retriever_instance is not None
        and retriever_instance.is_initialized
        and legal_service_instance is not None
    )
    indexed_count = len(retriever_instance.bm25_retriever.chunks) if is_ready else 0
    model_name = retriever_instance.model_name if is_ready else "BAAI/bge-m3"

    active_provider = "none"
    if qa_pipeline_instance is not None and qa_pipeline_instance.generator is not None:
        cfg = getattr(qa_pipeline_instance.generator, "config", None)
        active_provider = cfg.provider if cfg else "custom"

    return HealthResponse(
        status="ok" if is_ready else "initializing",
        version="1.0.0",
        retrieval_engine_initialized=is_ready,
        indexed_chunks=indexed_count,
        embedding_model=model_name,
        llm_provider=active_provider
    )


@app.post("/api/v1/search", response_model=SearchResponse, tags=["Retrieval"])
async def search_passages(req: SearchRequest):
    """
    Performs hybrid lexical (BM25) and semantic (BGE-M3) retrieval
    using Reciprocal Rank Fusion (RRF) across all ingested legal Acts.
    """
    if legal_service_instance is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Retrieval engine is not yet initialized."
        )

    return legal_service_instance.search_passages(query=req.query, top_k=req.top_k, act_filter=req.act_filter)


@app.post("/api/v1/civilian/ask", response_model=CivilianAskResponse, tags=["Civilian Portal"])
async def ask_civilian(req: BaseAskRequest):
    """
    Civilian endpoint: Returns plain-language framing, key takeaways,
    and official statutory citations with strict anti-hallucination guardrails.
    """
    if legal_service_instance is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Retrieval engine is not yet initialized."
        )

    return legal_service_instance.process_civilian_query(query=req.query, top_k=req.top_k, act_filter=req.act_filter)


@app.post("/api/v1/student/ask", response_model=StudentAskResponse, tags=["Student Portal"])
async def ask_student(req: BaseAskRequest):
    """
    Student endpoint: Returns statutory analysis, legislative scheme,
    essential ingredients, and pedagogical references without external hallucinations.
    """
    if legal_service_instance is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Retrieval engine is not yet initialized."
        )

    return legal_service_instance.process_student_query(query=req.query, top_k=req.top_k, act_filter=req.act_filter)


@app.post("/api/v1/lawyer/research", response_model=LawyerResearchResponse, tags=["Lawyer Portal"])
async def research_lawyer(req: LawyerResearchRequest):
    """
    Lawyer endpoint: Returns comprehensive statutory research dossier,
    verbatim statutory excerpts, source files, page ranges, and RRF scores.
    """
    if legal_service_instance is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Retrieval engine is not yet initialized."
        )

    return legal_service_instance.process_lawyer_query(
        query=req.query,
        top_k=req.top_k,
        act_filter=req.act_filter
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=False)
