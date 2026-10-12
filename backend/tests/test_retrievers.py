"""Tests for retrievers"""

import pytest
from app.core.retrievers.semantic import SemanticRetriever
from app.core.retrievers.bm25 import BM25Retriever
from app.core.retrievers.hybrid import HybridRetriever
from app.core.models import Document


@pytest.fixture
def sample_documents():
    return {
        "doc_1": Document(id="doc_1", content="Machine learning is a subset of artificial intelligence", metadata={"title": "ML"}),
        "doc_2": Document(id="doc_2", content="Deep learning uses neural networks with multiple layers", metadata={"title": "DL"}),
        "doc_3": Document(id="doc_3", content="Natural language processing helps computers understand human language", metadata={"title": "NLP"}),
        "doc_4": Document(id="doc_4", content="Computer vision trains computers to interpret visual world", metadata={"title": "CV"}),
    }


class TestSemanticRetriever:
    def test_index_and_retrieve(self, sample_documents):
        retriever = SemanticRetriever()
        retriever.index_documents(sample_documents)

        results = retriever.retrieve("machine learning", k=2)
        assert len(results) == 2
        assert results[0].document_id == "doc_1"
        assert results[0].score > 0

    def test_empty_index(self):
        retriever = SemanticRetriever()
        results = retriever.retrieve("test", k=5)
        assert results == []

    def test_retrieve_k_limit(self, sample_documents):
        retriever = SemanticRetriever()
        retriever.index_documents(sample_documents)

        results = retriever.retrieve("learning", k=2)
        assert len(results) == 2


class TestBM25Retriever:
    def test_index_and_retrieve(self, sample_documents):
        retriever = BM25Retriever()
        retriever.index_documents(sample_documents)

        results = retriever.retrieve("neural networks", k=2)
        assert len(results) == 2
        assert results[0].document_id == "doc_2"
        assert results[0].score > 0

    def test_empty_index(self):
        retriever = BM25Retriever()
        results = retriever.retrieve("test", k=5)
        assert results == []

    def test_exact_match(self, sample_documents):
        retriever = BM25Retriever()
        retriever.index_documents(sample_documents)

        results = retriever.retrieve("artificial intelligence", k=1)
        assert len(results) == 1


class TestHybridRetriever:
    def test_index_and_retrieve(self, sample_documents):
        retriever = HybridRetriever(semantic_weight=0.5)
        retriever.index_documents(sample_documents)

        results = retriever.retrieve("machine learning", k=2)
        assert len(results) == 2
        assert results[0].score > 0

    def test_weight_adjustment(self, sample_documents):
        retriever = HybridRetriever(semantic_weight=0.8)
        retriever.index_documents(sample_documents)

        results = retriever.retrieve("learning", k=2)
        assert len(results) == 2

    def test_empty_index(self):
        retriever = HybridRetriever()
        results = retriever.retrieve("test", k=5)
        assert results == []