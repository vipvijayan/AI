# RAG Application

A production-ready Retrieval-Augmented Generation (RAG) system with React frontend and Python backend, featuring hybrid retrieval (semantic + BM25) and cross-encoder reranking.

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              RAG APPLICATION ARCHITECTURE                        │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  ┌──────────────┐     HTTP/JSON      ┌──────────────────────────────────────┐  │
│  │   REACT      │ ◄─────────────────► │         FASTAPI BACKEND              │  │
│  │  FRONTEND    │     REST API        │                                      │  │
│  │  (Port 3000) │                     │  ┌────────────────────────────────┐  │  │
│  └──────────────┘                     │  │        RAG PIPELINE            │  │  │
│                                       │  │                                │  │  │
│  ┌──────────────────────────────────┐ │  │  ┌──────────┐ ┌────────────┐  │  │  │
│  │        USER INTERFACE            │ │  │  │  QUERY   │ │  RETRIEVER │  │  │  │
│  │                                  │ │  │  │ ANALYZER │ │  (HYBRID)  │  │  │  │
│  │  ┌────────────────────────────┐  │ │  │  └────┬─────┘ └─────┬──────┘  │  │  │
│  │  │   Search Input             │  │ │  │       │           │          │  │  │
│  │  │   Options (Optimize,       │  │ │  │       ▼           ▼          │  │  │
│  │  │    Rerank, Top-K)          │  │ │  │  ┌────────────────────────┐   │  │  │
│  │  └────────────────────────────┘  │ │  │  │    QUERY OPTIMIZER     │   │  │  │
│  │                                  │ │  │  │  (Expand, Rewrite,     │   │  │  │
│  │  ┌────────────────────────────┐  │ │  │  │   Stop-word Removal)   │   │  │  │
│  │  │   Results Display          │  │ │  │  └───────────┬────────────┘   │  │  │
│  │  │   - Ranked Results         │  │ │  │              │                │  │  │
│  │  │   - Relevance Scores       │  │ │  │              ▼                │  │  │
│  │  │   - Document Metadata      │  │ │  │  ┌────────────────────────┐   │  │  │
│  │  └────────────────────────────┘  │ │  │  │   HYBRID RETRIEVER     │   │  │  │
│  └──────────────────────────────────┘ │  │  │  ┌─────────┬─────────┐   │   │  │  │
│                                       │  │  │  │SEMANTIC │  BM25   │   │   │  │  │
│                                       │  │  │  │(Embed-  │ (Lexical)│   │   │  │  │
│                                       │  │  │  │ dings)  │          │   │   │  │  │
│                                       │  │  │  └────┬────┴────┬────┘   │   │  │  │
│                                       │  │  │       │         │          │   │  │  │
│                                       │  │  │       ▼         ▼          │   │  │  │
│                                       │  │  │  ┌────────────────────────┐  │  │  │
│                                       │  │  │  │   RRF FUSION (k=60)    │  │  │  │
│                                       │  │  │  │  1/(k + rank)          │  │  │  │
│                                       │  │  │  └───────────┬────────────┘  │  │  │
│                                       │  │  │              │               │  │  │
│                                       │  │  │              ▼               │  │  │
│                                       │  │  │  ┌────────────────────────┐  │  │  │
│                                       │  │  │  │  CROSS-ENCODER         │  │  │  │
│                                       │  │  │  │  RERANKER              │  │  │  │
│                                       │  │  │  │  (ms-marco-MiniLM-L-6) │  │  │  │
│                                       │  │  │  └───────────┬────────────┘  │  │  │
│                                       │  │  │              │               │  │  │
│                                       │  │  │              ▼               │  │  │
│                                       │  │  │  ┌────────────────────────┐  │  │  │
│                                       │  │  │  │    TOP-K RESULTS       │  │  │  │
│                                       │  │  │  └────────────────────────┘  │  │  │
│                                       │  │  └────────────────────────────┘  │  │  │
│                                       │  │                                  │  │  │
│                                       │  │  ┌────────────────────────────┐  │  │  │
│                                       │  │  │      DATA LAYER            │  │  │  │
│                                       │  │  │  ┌──────────────────────┐  │  │  │  │
│                                       │  │  │  │  MINI-WIKIPEDIA      │  │  │  │  │
│                                       │  │  │  │  DATASET             │  │  │  │  │
│                                       │  │  │  │  - 500 Documents     │  │  │  │  │
│                                       │  │  │  │  - Text Corpus       │  │  │  │  │
│                                       │  │  │  │  - QA Pairs          │  │  │  │  │
│                                       │  │  │  └──────────────────────┘  │  │  │  │
│                                       │  │  └────────────────────────────┘  │  │  │
│                                       │  └──────────────────────────────────┘  │  │
│                                       │            (Port 8000)                │  │
│                                       └──────────────────────────────────────┘  │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

## Data Flow

```
┌─────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│   USER      │───►│  QUERY       │───►│  QUERY       │───►│  HYBRID      │
│   QUERY     │    │  ANALYSIS    │    │  OPTIMIZER   │    │  RETRIEVER   │
└─────────────┘    └──────────────┘    └──────────────┘    └──────┬───────┘
                                                                    │
                                                    ┌───────────────┴───────────────┐
                                                    ▼                               ▼
                                           ┌──────────────┐                 ┌──────────────┐
                                           │  SEMANTIC    │                 │    BM25      │
                                           │  RETRIEVER   │                 │  RETRIEVER   │
                                           │  (Embeddings)│                 │  (Lexical)   │
                                           └──────┬───────┘                 └──────┬───────┘
                                                  │                              │
                                                  └──────────────┬───────────────┘
                                                                 ▼
                                                        ┌──────────────┐
                                                        │  RRF FUSION  │
                                                        │  (k=60)      │
                                                        └──────┬───────┘
                                                               │
                                                               ▼
                                                        ┌──────────────┐
                                                        │  CROSS-      │
                                                        │  ENCODER     │
                                                        │  RERANKER    │
                                                        └──────┬───────┘
                                                               │
                                                               ▼
                                                        ┌──────────────┐
                                                        │   TOP-K      │
                                                        │  RESULTS     │
                                                        └──────────────┘
```

## Features

- **Hybrid Retrieval**: Combines semantic (embeddings) and lexical (BM25) search using Reciprocal Rank Fusion
- **Cross-Encoder Reranking**: Uses `cross-encoder/ms-marco-MiniLM-L-6-v2` for relevance scoring
- **Query Optimization**: Query expansion, rewriting, and stop-word removal
- **Query Analysis**: Intent detection, keyword extraction, entity recognition
- **Mini-Wikipedia Dataset**: Pre-loaded with 500 documents from `rag-datasets/rag-mini-wikipedia`
- **RESTful API**: FastAPI backend with automatic OpenAPI documentation
- **Modern Frontend**: React 18 + TypeScript + Vite
- **Containerized**: Docker Compose for easy deployment
- **Testable**: Comprehensive test suites for both backend and frontend

## Quick Start

### Using Docker Compose (Recommended)

```bash
cd ai_rag
docker-compose up --build
```

- Frontend: http://localhost:3000
- Backend API: http://localhost:8000
- API Docs: http://localhost:8000/docs

### Manual Development Setup

#### Backend

```bash
cd ai_rag/backend
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt
uvicorn app.main:app --reload
```

#### Frontend

```bash
cd ai_rag/frontend
npm install
npm run dev
```

## Project Structure

```
ai_rag/
├── backend/
│   ├── app/
│   │   ├── api/
│   │   │   └── routes.py          # FastAPI routes
│   │   ├── core/
│   │   │   ├── models.py          # Data models
│   │   │   ├── pipeline.py        # RAG pipeline orchestrator
│   │   │   ├── reranker.py        # Cross-encoder reranker
│   │   │   ├── retrievers/
│   │   │   │   ├── base.py        # Base retriever interface
│   │   │   │   ├── semantic.py    # Semantic retrieval (embeddings)
│   │   │   │   ├── bm25.py        # BM25 lexical retrieval
│   │   │   │   └── hybrid.py      # Hybrid retrieval (RRF)
│   │   │   └── query/
│   │   │       ├── analyzer.py    # Query analysis
│   │   │       └── optimizer.py   # Query optimization
│   │   ├── data/
│   │   │   └── loader.py          # Wikipedia data loader
│   │   └── main.py                # FastAPI app factory
│   ├── tests/                     # Backend tests
│   ├── requirements.txt
│   └── Dockerfile
├── frontend/
│   ├── src/
│   │   ├── api/
│   │   │   └── client.ts          # API client
│   │   ├── components/            # React components
│   │   ├── hooks/                 # Custom hooks
│   │   ├── types/                 # TypeScript types
│   │   ├── App.tsx                # Main app component
│   │   └── main.tsx               # Entry point
│   ├── tests/                     # Frontend tests
│   ├── package.json
│   ├── vite.config.ts
│   └── Dockerfile
├── docker-compose.yml
└── README.md
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/v1/query` | Execute RAG query |
| GET | `/api/v1/query` | Execute RAG query (GET) |
| GET | `/api/v1/stats` | Get pipeline statistics |
| GET | `/api/v1/health` | Health check |

### Query Parameters

```json
{
  "query": "string",
  "k": 10,
  "optimize_query": true,
  "rerank": true,
  "rerank_k": 5
}
```

### Response Format

```json
{
  "query": "string",
  "results": [
    {
      "document_id": "string",
      "content": "string",
      "score": 0.95,
      "metadata": {}
    }
  ],
  "total_results": 5,
  "processing_time_ms": 150.5
}
```

## Running Tests

### Backend Tests

```bash
cd ai_rag/backend
pytest -v
```

### Frontend Tests

```bash
cd ai_rag/frontend
npm test
```

## Configuration

### Backend Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MAX_DOCUMENTS` | 500 | Number of documents to load |
| `SEMANTIC_MODEL` | all-MiniLM-L6-v2 | Sentence transformer model |
| `RERANKER_MODEL` | cross-encoder/ms-marco-MiniLM-L-6-v2 | Cross-encoder model |
| `SEMANTIC_WEIGHT` | 0.5 | Weight for semantic retrieval |
| `K_RRF` | 60 | RRF fusion parameter |

### Frontend Configuration

Edit `vite.config.ts` to change proxy target or port.

## Extending the System

### Adding a New Retriever

1. Create a new class in `backend/app/core/retrievers/` extending `BaseRetriever`
2. Implement `_build_index()` and `retrieve()` methods
3. Register in the pipeline

### Adding a New Query Optimizer

1. Extend `QueryOptimizer` in `backend/app/core/query/optimizer.py`
2. Add new strategies to the `optimize()` method

### Using Different Datasets

Modify `backend/app/data/loader.py` to load custom datasets.

## Performance Considerations

- **Embedding Model**: `all-MiniLM-L6-v2` (384 dimensions, fast inference)
- **Reranker Model**: `cross-encoder/ms-marco-MiniLM-L-6-v2` (high quality, moderate speed)
- **Index Size**: 500 documents by default, adjustable via `MAX_DOCUMENTS`
- **Retrieval K**: Default 10, rerank to top 5
- **RRF Parameter**: k=60 (standard value)

## License

MIT