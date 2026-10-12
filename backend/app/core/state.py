"""Global application state"""

from typing import Optional
from app.core.pipeline import RAGPipeline


class PipelineState:
    """Manages the global RAG pipeline instance"""

    def __init__(self):
        self._pipeline: Optional[RAGPipeline] = None

    def set_pipeline(self, pipeline: RAGPipeline):
        self._pipeline = pipeline

    def get_pipeline(self) -> Optional[RAGPipeline]:
        return self._pipeline

    def is_ready(self) -> bool:
        return self._pipeline is not None


rag_pipeline = PipelineState()