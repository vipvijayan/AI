"""Main RAG pipeline orchestrator"""

from typing import List, Dict, Optional
import time
from app.core.retrievers.base import BaseRetriever
from app.core.reranker import CrossEncoderReranker
from app.core.query.analyzer import QueryAnalyzer
from app.core.query.optimizer import QueryOptimizer
from app.core.models import RetrievalResult, Document, RAGResponse


class RAGPipeline:
    """Production RAG Pipeline"""

    def __init__(
        self,
        retriever: BaseRetriever,
        reranker: Optional[CrossEncoderReranker] = None,
        query_optimizer: Optional[QueryOptimizer] = None,
        query_analyzer: Optional[QueryAnalyzer] = None,
    ):
        self.retriever = retriever
        self.reranker = reranker or CrossEncoderReranker()
        self.query_optimizer = query_optimizer or QueryOptimizer()
        self.query_analyzer = query_analyzer or QueryAnalyzer()

        self.retrieval_stats = {
            "queries_processed": 0,
            "total_retrieval_time": 0.0,
            "total_reranking_time": 0.0,
        }

    def index_documents(self, documents: Dict[str, Document]):
        """Index documents for retrieval"""
        self.retriever.index_documents(documents)

    def query(
        self,
        query: str,
        k: int = 10,
        optimize_query: bool = True,
        rerank: bool = True,
        rerank_k: int = 5,
    ) -> RAGResponse:
        """Execute RAG query"""
        start_time = time.time()
        self.retrieval_stats["queries_processed"] += 1

        # Handle empty queries
        if not query or not query.strip():
            return RAGResponse(
                query=query,
                results=[],
                total_results=0,
                processing_time_ms=(time.time() - start_time) * 1000,
            )

        analysis = self.query_analyzer.analyze(query)

        queries_to_run = [query]
        if optimize_query:
            strategy = self.query_analyzer.suggest_strategy(query)
            queries_to_run = self.query_optimizer.optimize(query, strategy)

        all_results: Dict[str, RetrievalResult] = {}

        for q in queries_to_run:
            results = self.retriever.retrieve(q, k=k)

            for result in results:
                if result.document_id not in all_results:
                    all_results[result.document_id] = result
                else:
                    existing = all_results[result.document_id]
                    avg_score = (existing.score + result.score) / 2
                    all_results[result.document_id].score = avg_score

        results_list = list(all_results.values())
        results_list.sort(key=lambda x: x.score, reverse=True)
        results_list = results_list[:k]

        retrieval_time = time.time() - start_time
        self.retrieval_stats["total_retrieval_time"] += retrieval_time

        if rerank and results_list:
            rerank_start = time.time()
            results_list = self.reranker.rerank(query, results_list, k=rerank_k)
            rerank_time = time.time() - rerank_start
            self.retrieval_stats["total_reranking_time"] += rerank_time

        total_time = time.time() - start_time

        return RAGResponse(
            query=query,
            results=results_list,
            total_results=len(results_list),
            processing_time_ms=total_time * 1000,
        )

    def get_stats(self) -> Dict:
        """Get retrieval statistics"""
        return {
            **self.retrieval_stats,
            "retriever": self.retriever.name,
            "documents_indexed": len(self.retriever.documents),
        }