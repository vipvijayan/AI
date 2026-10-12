"""FastAPI application factory"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from app.api.routes import router as api_router
from app.core.pipeline import RAGPipeline
from app.core.retrievers.hybrid import HybridRetriever
from app.core.reranker import CrossEncoderReranker
from app.core.query.analyzer import QueryAnalyzer
from app.core.query.optimizer import QueryOptimizer
from app.data.loader import WikipediaDataLoader
from app.core.models import Document
from app.core.state import rag_pipeline


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Starting RAG application...")
    print("Loading dataset...")

    loader = WikipediaDataLoader(max_documents=500)
    documents = loader.load()

    retriever = HybridRetriever(semantic_weight=0.5)
    reranker = CrossEncoderReranker()
    query_optimizer = QueryOptimizer()
    query_analyzer = QueryAnalyzer()

    rag_pipeline.set_pipeline(RAGPipeline(
        retriever=retriever,
        reranker=reranker,
        query_optimizer=query_optimizer,
        query_analyzer=query_analyzer,
    ))

    rag_pipeline.get_pipeline().index_documents(documents)
    print(f"Indexed {len(documents)} documents")
    print("RAG pipeline ready!")

    yield

    print("Shutting down...")


def create_app() -> FastAPI:
    app = FastAPI(
        title="RAG Application",
        description="Production RAG system with hybrid retrieval and reranking",
        version="1.0.0",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(api_router, prefix="/api/v1")

    return app


app = create_app()