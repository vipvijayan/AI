"""Base retriever interface"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any
from app.core.models import RetrievalResult, Document


class BaseRetriever(ABC):
    """Abstract base class for all retrieval strategies"""

    def __init__(self, name: str = None):
        self.name = name or self.__class__.__name__
        self.documents: Dict[str, Document] = {}

    @abstractmethod
    def retrieve(self, query: str, k: int = 10) -> List[RetrievalResult]:
        """Retrieve top-k documents for query"""
        pass

    def index_documents(self, documents: Dict[str, Document]):
        """Index documents for retrieval"""
        self.documents = documents
        self._build_index()

    @abstractmethod
    def _build_index(self):
        """Build index from documents"""
        pass

    def get_document(self, doc_id: str) -> Document:
        """Get document by ID"""
        return self.documents.get(doc_id)

    def get_stats(self) -> Dict[str, Any]:
        """Get retriever statistics"""
        return {
            "name": self.name,
            "document_count": len(self.documents),
        }

    def __repr__(self):
        return f"{self.name}(documents={len(self.documents)})"