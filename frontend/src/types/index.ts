export interface RetrievalResult {
  document_id: string
  content: string
  score: number
  metadata: Record<string, unknown>
}

export interface QueryResponse {
  query: string
  results: RetrievalResult[]
  total_results: number
  processing_time_ms: number
}

export interface StatsResponse {
  queries_processed: number
  total_retrieval_time: number
  total_reranking_time: number
  retriever: string
  documents_indexed: number
}

export interface QueryParams {
  query: string
  k?: number
  optimize_query?: boolean
  rerank?: boolean
  rerank_k?: number
}