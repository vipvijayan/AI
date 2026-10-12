import axios from 'axios'

const api = axios.create({
  baseURL: '/api/v1',
  timeout: 30000,
})

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

export async function queryRag(params: QueryParams): Promise<QueryResponse> {
  const response = await api.post('/query', params)
  return response.data
}

export async function getStats(): Promise<StatsResponse> {
  const response = await api.get('/stats')
  return response.data
}

export async function healthCheck(): Promise<{ status: string; pipeline_ready: boolean }> {
  const response = await api.get('/health')
  return response.data
}