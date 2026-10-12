"""Core data models for RAG system"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional


@dataclass
class Document:
    """Document model"""
    id: str
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RetrievalResult:
    """Single retrieval result"""
    document_id: str
    content: str
    score: float
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "document_id": self.document_id,
            "content": self.content,
            "score": self.score,
            "metadata": self.metadata,
        }


@dataclass
class QueryAnalysis:
    """Result of query analysis"""
    original: str
    cleaned: str
    intent: str
    keywords: List[str]
    entities: List[str]
    length: int
    is_question: bool


@dataclass
class RAGResponse:
    """RAG query response"""
    query: str
    results: List[RetrievalResult]
    total_results: int
    processing_time_ms: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "results": [r.to_dict() for r in self.results],
            "total_results": self.total_results,
            "processing_time_ms": self.processing_time_ms,
        }


@dataclass
class IndexStats:
    """Index statistics"""
    document_count: int
    retriever_type: str
    indexed: bool