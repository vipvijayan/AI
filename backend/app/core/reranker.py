"""Cross-encoder reranking for improving retrieval quality"""

from typing import List
from sentence_transformers import CrossEncoder
from app.core.models import RetrievalResult


class CrossEncoderReranker:
    """Reranking using cross-encoder model"""

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model = CrossEncoder(model_name)

    def rerank(self, query: str, candidates: List[RetrievalResult], k: int = None) -> List[RetrievalResult]:
        """Rerank candidates using cross-encoder model"""
        if not candidates:
            return []

        pairs = [(query, result.content) for result in candidates]
        scores = self.model.predict(pairs)

        scored = list(zip(candidates, scores))
        scored.sort(key=lambda x: x[1], reverse=True)

        reranked = []
        for result, relevance in scored:
            reranked_result = RetrievalResult(
                document_id=result.document_id,
                content=result.content,
                score=float(relevance),
                metadata={
                    **result.metadata,
                    "original_score": result.score,
                    "reranker_score": float(relevance),
                }
            )
            reranked.append(reranked_result)

        return reranked[:k] if k else reranked