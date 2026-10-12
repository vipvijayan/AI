"""API routes for RAG application"""

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
from typing import List, Optional

from app.core.pipeline import RAGPipeline
from app.core.models import RAGResponse, RetrievalResult
from app.main import rag_pipeline


router = APIRouter()


class QueryRequest(BaseModel):
    query: str
    k: int = 10
    optimize_query: bool = True
    rerank: bool = True
    rerank_k: int = 5


class QueryResponse(BaseModel):
    query: str
    results: List[dict]
    total_results: int
    processing_time_ms: float


class StatsResponse(BaseModel):
    queries_processed: int
    total_retrieval_time: float
    total_reranking_time: float
    retriever: str
    documents_indexed: int


@router.post("/query", response_model=QueryResponse)
async def query_rag(request: QueryRequest):
    """Execute RAG query"""
    if not rag_pipeline:
        raise HTTPException(status_code=503, detail="RAG pipeline not initialized")

    response: RAGResponse = rag_pipeline.query(
        query=request.query,
        k=request.k,
        optimize_query=request.optimize_query,
        rerank=request.rerank,
        rerank_k=request.rerank_k,
    )

    return QueryResponse(
        query=response.query,
        results=[r.to_dict() for r in response.results],
        total_results=response.total_results,
        processing_time_ms=response.processing_time_ms,
    )


@router.get("/query", response_model=QueryResponse)
async def query_rag_get(
    q: str = Query(..., description="Query string"),
    k: int = Query(10, ge=1, le=50),
    optimize_query: bool = Query(True),
    rerank: bool = Query(True),
    rerank_k: int = Query(5, ge=1, le=20),
):
    """Execute RAG query via GET"""
    if not rag_pipeline:
        raise HTTPException(status_code=503, detail="RAG pipeline not initialized")

    response: RAGResponse = rag_pipeline.query(
        query=q,
        k=k,
        optimize_query=optimize_query,
        rerank=rerank,
        rerank_k=rerank_k,
    )

    return QueryResponse(
        query=response.query,
        results=[r.to_dict() for r in response.results],
        total_results=response.total_results,
        processing_time_ms=response.processing_time_ms,
    )


@router.get("/stats", response_model=StatsResponse)
async def get_stats():
    """Get pipeline statistics"""
    if not rag_pipeline:
        raise HTTPException(status_code=503, detail="RAG pipeline not initialized")

    stats = rag_pipeline.get_stats()
    return StatsResponse(**stats)


@router.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "pipeline_ready": rag_pipeline is not None,
    }