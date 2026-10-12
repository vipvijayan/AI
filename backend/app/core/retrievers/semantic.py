"""Semantic retrieval using embeddings"""

from typing import List, Dict
import numpy as np
from sentence_transformers import SentenceTransformer
from app.core.retrievers.base import BaseRetriever
from app.core.models import RetrievalResult, Document


class SemanticRetriever(BaseRetriever):
    """Semantic retrieval using sentence embeddings"""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", embedding_dim: int = 384):
        super().__init__(name="SemanticRetriever")
        self.model_name = model_name
        self.embedding_dim = embedding_dim
        self.model = SentenceTransformer(model_name)
        self.doc_embeddings: Dict[str, np.ndarray] = {}

    def _build_index(self):
        """Generate embeddings for all documents"""
        self.doc_embeddings = {}
        for doc_id, doc in self.documents.items():
            self.doc_embeddings[doc_id] = self.model.encode(doc.content, normalize_embeddings=True)

    def retrieve(self, query: str, k: int = 10) -> List[RetrievalResult]:
        """Retrieve documents using semantic similarity"""
        if not self.documents:
            return []

        query_embedding = self.model.encode(query, normalize_embeddings=True)

        scores = []
        for doc_id, doc_embedding in self.doc_embeddings.items():
            similarity = float(np.dot(query_embedding, doc_embedding))
            scores.append((doc_id, similarity))

        scores.sort(key=lambda x: x[1], reverse=True)

        results = []
        for doc_id, score in scores[:k]:
            doc = self.documents[doc_id]
            results.append(RetrievalResult(
                document_id=doc_id,
                content=doc.content,
                score=score,
                metadata=doc.metadata
            ))

        return results