"""Tests for RAG pipeline"""

import pytest
from app.core.pipeline import RAGPipeline
from app.core.retrievers.hybrid import HybridRetriever
from app.core.reranker import CrossEncoderReranker
from app.core.query.analyzer import QueryAnalyzer
from app.core.query.optimizer import QueryOptimizer
from app.core.models import Document


@pytest.fixture
def sample_documents():
    return {
        "doc_1": Document(id="doc_1", content="Machine learning is a subset of artificial intelligence that enables computers to learn from data", metadata={"title": "ML"}),
        "doc_2": Document(id="doc_2", content="Deep learning uses neural networks with multiple layers to progressively extract features", metadata={"title": "DL"}),
        "doc_3": Document(id="doc_3", content="Natural language processing helps computers understand and generate human language", metadata={"title": "NLP"}),
        "doc_4": Document(id="doc_4", content="Computer vision enables computers to interpret and understand visual information", metadata={"title": "CV"}),
        "doc_5": Document(id="doc_5", content="Artificial intelligence encompasses machine learning, deep learning, and other techniques", metadata={"title": "AI"}),
    }


@pytest.fixture
def pipeline(sample_documents):
    retriever = HybridRetriever(semantic_weight=0.5)
    reranker = CrossEncoderReranker()
    query_optimizer = QueryOptimizer()
    query_analyzer = QueryAnalyzer()

    pipeline = RAGPipeline(
        retriever=retriever,
        reranker=reranker,
        query_optimizer=query_optimizer,
        query_analyzer=query_analyzer,
    )
    pipeline.index_documents(sample_documents)
    return pipeline


class TestRAGPipeline:
    def test_query_basic(self, pipeline):
        response = pipeline.query("What is machine learning?", k=3)
        assert response.query == "What is machine learning?"
        assert response.total_results > 0
        assert response.processing_time_ms > 0
        assert len(response.results) <= 3

    def test_query_with_rerank(self, pipeline):
        response = pipeline.query("neural networks", k=5, rerank=True, rerank_k=3)
        assert response.total_results <= 3
        for result in response.results:
            assert "reranker_score" in result.metadata

    def test_query_without_optimize(self, pipeline):
        response = pipeline.query("deep learning", k=3, optimize_query=False)
        assert response.total_results > 0

    def test_query_without_rerank(self, pipeline):
        response = pipeline.query("NLP", k=3, rerank=False)
        assert response.total_results > 0
        for result in response.results:
            assert "reranker_score" not in result.metadata or result.metadata.get("reranker_score") is None

    def test_get_stats(self, pipeline):
        pipeline.query("test query", k=3)
        stats = pipeline.get_stats()
        assert stats["queries_processed"] == 1
        assert stats["retriever"] == "HybridRetriever"
        assert stats["documents_indexed"] == 5

    def test_multiple_queries(self, pipeline):
        queries = ["machine learning", "neural networks", "computer vision"]
        for q in queries:
            response = pipeline.query(q, k=2)
            assert response.total_results > 0

        stats = pipeline.get_stats()
        assert stats["queries_processed"] == 3

    def test_empty_query(self, pipeline):
        response = pipeline.query("", k=3)
        assert response.total_results == 0