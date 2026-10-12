import { describe, it, expect, vi, beforeEach } from 'vitest'

vi.mock('axios', () => ({
  default: {
    create: vi.fn(() => ({
      post: vi.fn(),
      get: vi.fn(),
    })),
  },
}))

import axios from 'axios'
import { queryRag, getStats, healthCheck } from '../api/client'

const mockPost = vi.mocked(axios.create).mock.results[0]?.value?.post as ReturnType<typeof vi.fn>
const mockGet = vi.mocked(axios.create).mock.results[0]?.value?.get as ReturnType<typeof vi.fn>

describe('API Client', () => {
  beforeEach(() => {
    vi.clearAllMocks()
  })

  describe('queryRag', () => {
    it('should call POST /query with provided params', async () => {
      const mockResponse = {
        data: {
          query: 'test query',
          results: [
            { document_id: 'doc_1', content: 'Test content', score: 0.9, metadata: {} }
          ],
          total_results: 1,
          processing_time_ms: 150,
        },
      }
      mockPost.mockResolvedValue(mockResponse)

      const result = await queryRag({ query: 'test query', k: 10 })

      expect(mockPost).toHaveBeenCalledWith('/query', {
        query: 'test query',
        k: 10,
      })
      expect(result.query).toBe('test query')
      expect(result.results).toHaveLength(1)
    })

    it('should handle optional parameters when provided', async () => {
      const mockResponse = {
        data: {
          query: 'test',
          results: [],
          total_results: 0,
          processing_time_ms: 50,
        },
      }
      mockPost.mockResolvedValue(mockResponse)

      await queryRag({ query: 'test', k: 5, optimize_query: false, rerank: false, rerank_k: 3 })

      expect(mockPost).toHaveBeenCalledWith('/query', {
        query: 'test',
        k: 5,
        optimize_query: false,
        rerank: false,
        rerank_k: 3,
      })
    })
  })

  describe('getStats', () => {
    it('should call GET /stats', async () => {
      const mockResponse = {
        data: {
          queries_processed: 10,
          total_retrieval_time: 1.5,
          total_reranking_time: 0.5,
          retriever: 'HybridRetriever',
          documents_indexed: 100,
        },
      }
      mockGet.mockResolvedValue(mockResponse)

      const result = await getStats()

      expect(mockGet).toHaveBeenCalledWith('/stats')
      expect(result.queries_processed).toBe(10)
      expect(result.retriever).toBe('HybridRetriever')
    })
  })

  describe('healthCheck', () => {
    it('should call GET /health', async () => {
      const mockResponse = {
        data: { status: 'healthy', pipeline_ready: true },
      }
      mockGet.mockResolvedValue(mockResponse)

      const result = await healthCheck()

      expect(mockGet).toHaveBeenCalledWith('/health')
      expect(result.status).toBe('healthy')
      expect(result.pipeline_ready).toBe(true)
    })
  })
})