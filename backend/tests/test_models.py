"""Tests for core models"""

import pytest
from app.core.models import Document, RetrievalResult, QueryAnalysis, RAGResponse, IndexStats


class TestDocument:
    def test_create_document(self):
        doc = Document(id="test_1", content="Test content", metadata={"key": "value"})
        assert doc.id == "test_1"
        assert doc.content == "Test content"
        assert doc.metadata == {"key": "value"}

    def test_create_document_default_metadata(self):
        doc = Document(id="test_1", content="Test content")
        assert doc.metadata == {}


class TestRetrievalResult:
    def test_create_result(self):
        result = RetrievalResult(
            document_id="doc_1",
            content="Test content",
            score=0.95,
            metadata={"title": "Test"},
        )
        assert result.document_id == "doc_1"
        assert result.score == 0.95

    def test_to_dict(self):
        result = RetrievalResult(
            document_id="doc_1",
            content="Test content",
            score=0.95,
            metadata={"title": "Test"},
        )
        d = result.to_dict()
        assert d["document_id"] == "doc_1"
        assert d["score"] == 0.95
        assert d["metadata"]["title"] == "Test"


class TestQueryAnalysis:
    def test_create_analysis(self):
        analysis = QueryAnalysis(
            original="What is AI?",
            cleaned="what is ai",
            intent="definition",
            keywords=["what", "ai"],
            entities=[],
            length=3,
            is_question=True,
        )
        assert analysis.intent == "definition"
        assert analysis.is_question is True


class TestRAGResponse:
    def test_create_response(self):
        results = [
            RetrievalResult(document_id="doc_1", content="Content 1", score=0.9),
            RetrievalResult(document_id="doc_2", content="Content 2", score=0.8),
        ]
        response = RAGResponse(
            query="test query",
            results=results,
            total_results=2,
            processing_time_ms=150.5,
        )
        assert response.query == "test query"
        assert response.total_results == 2

    def test_to_dict(self):
        results = [
            RetrievalResult(document_id="doc_1", content="Content 1", score=0.9),
        ]
        response = RAGResponse(
            query="test query",
            results=results,
            total_results=1,
            processing_time_ms=150.5,
        )
        d = response.to_dict()
        assert d["query"] == "test query"
        assert len(d["results"]) == 1
        assert d["processing_time_ms"] == 150.5


class TestIndexStats:
    def test_create_stats(self):
        stats = IndexStats(document_count=100, retriever_type="HybridRetriever", indexed=True)
        assert stats.document_count == 100
        assert stats.retriever_type == "HybridRetriever"
        assert stats.indexed is True