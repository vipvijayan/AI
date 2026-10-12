"""BM25 retrieval using lexical matching"""

from typing import List, Dict, Set
from collections import defaultdict
import math
from app.core.retrievers.base import BaseRetriever
from app.core.models import RetrievalResult, Document


class BM25Retriever(BaseRetriever):
    """BM25 (Best Matching 25) retrieval algorithm"""

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        super().__init__(name="BM25Retriever")
        self.k1 = k1
        self.b = b

        self.vocabulary: Set[str] = set()
        self.doc_freq: Dict[str, int] = defaultdict(int)
        self.term_positions: Dict[str, Dict[str, List[int]]] = defaultdict(dict)
        self.doc_lengths: Dict[str, int] = {}
        self.avg_doc_length: float = 0

    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization"""
        return text.lower().split()

    def _build_index(self):
        """Build BM25 index from documents"""
        self.vocabulary = set()
        self.doc_freq = defaultdict(int)
        self.term_positions = defaultdict(dict)
        self.doc_lengths = {}

        total_length = 0

        for doc_id, doc in self.documents.items():
            tokens = self._tokenize(doc.content)
            self.doc_lengths[doc_id] = len(tokens)
            total_length += len(tokens)

            doc_terms = set()

            for pos, token in enumerate(tokens):
                self.vocabulary.add(token)

                if doc_id not in self.term_positions[token]:
                    self.term_positions[token][doc_id] = []
                self.term_positions[token][doc_id].append(pos)

                doc_terms.add(token)

            for term in doc_terms:
                self.doc_freq[term] += 1

        num_docs = len(self.documents)
        self.avg_doc_length = total_length / max(num_docs, 1)

    def _bm25_score(self, query_tokens: List[str], doc_id: str) -> float:
        """Calculate BM25 score for document"""
        score = 0.0
        doc_length = self.doc_lengths.get(doc_id, 0)

        for term in set(query_tokens):
            if term not in self.term_positions or doc_id not in self.term_positions[term]:
                continue

            term_freq = len(self.term_positions[term][doc_id])

            num_docs = len(self.documents)
            doc_count = self.doc_freq.get(term, 0)
            idf = math.log((num_docs - doc_count + 0.5) / (doc_count + 0.5) + 1.0)

            norm_factor = 1 - self.b + self.b * (doc_length / max(self.avg_doc_length, 1))
            numerator = term_freq * (self.k1 + 1)
            denominator = term_freq + self.k1 * norm_factor

            score += idf * (numerator / denominator)

        return score

    def retrieve(self, query: str, k: int = 10) -> List[RetrievalResult]:
        """Retrieve documents using BM25 scoring"""
        if not self.documents:
            return []

        query_tokens = self._tokenize(query)
        if not query_tokens:
            return []

        scores = []
        for doc_id in self.documents:
            score = self._bm25_score(query_tokens, doc_id)
            scores.append((doc_id, score))

        scores.sort(key=lambda x: x[1], reverse=True)

        results = []
        for doc_id, score in scores[:k]:
            doc = self.documents[doc_id]
            results.append(RetrievalResult(
                document_id=doc_id,
                content=doc.content,
                score=float(score),
                metadata=doc.metadata
            ))

        return results