import { describe, it, expect, vi, beforeEach } from 'vitest'
import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import App from './App'
import * as apiClient from './api/client'

vi.mock('./api/client', () => ({
  queryRag: vi.fn(),
  getStats: vi.fn(),
  healthCheck: vi.fn(),
}))

describe('App', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    vi.mocked(apiClient.healthCheck).mockResolvedValue({ status: 'healthy', pipeline_ready: true })
    vi.mocked(apiClient.getStats).mockResolvedValue({
      queries_processed: 0,
      total_retrieval_time: 0,
      total_reranking_time: 0,
      retriever: 'HybridRetriever',
      documents_indexed: 100,
    })
  })

  it('renders header', () => {
    render(<App />)
    expect(screen.getByText('RAG Application')).toBeInTheDocument()
    expect(screen.getByText('Hybrid Retrieval with Cross-Encoder Reranking')).toBeInTheDocument()
  })

  it('shows empty state initially', () => {
    render(<App />)
    expect(screen.getByText('Welcome to RAG Application')).toBeInTheDocument()
    expect(screen.getByText('Enter a question above to search the knowledge base')).toBeInTheDocument()
  })

  it('searches and displays results', async () => {
    const mockResults = {
      query: 'machine learning',
      results: [
        { document_id: 'doc_1', content: 'Machine learning is a subset of AI', score: 0.95, metadata: { title: 'ML' } },
        { document_id: 'doc_2', content: 'Deep learning uses neural networks', score: 0.85, metadata: { title: 'DL' } },
      ],
      total_results: 2,
      processing_time_ms: 150,
    }
    vi.mocked(apiClient.queryRag).mockResolvedValue(mockResults)

    render(<App />)

    const input = screen.getByPlaceholderText('Ask a question...')
    const button = screen.getByRole('button', { name: /search/i })

    await userEvent.type(input, 'machine learning')
    await userEvent.click(button)

    await waitFor(() => {
      expect(screen.getByText('Machine learning is a subset of AI')).toBeInTheDocument()
    })

    expect(screen.getByText('Deep learning uses neural networks')).toBeInTheDocument()
    expect(screen.getByText('#1')).toBeInTheDocument()
    expect(screen.getByText('#2')).toBeInTheDocument()
  })

  it('shows error when query fails', async () => {
    vi.mocked(apiClient.queryRag).mockRejectedValue(new Error('Network error'))

    render(<App />)

    const input = screen.getByPlaceholderText('Ask a question...')
    const button = screen.getByRole('button', { name: /search/i })

    await userEvent.type(input, 'test query')
    await userEvent.click(button)

    await waitFor(() => {
      expect(screen.getByText('Failed to execute query. Please try again.')).toBeInTheDocument()
    })
  })

  it('clears results when clear button clicked', async () => {
    const mockResults = {
      query: 'test',
      results: [{ document_id: 'doc_1', content: 'Test content', score: 0.9, metadata: {} }],
      total_results: 1,
      processing_time_ms: 100,
    }
    vi.mocked(apiClient.queryRag).mockResolvedValue(mockResults)

    render(<App />)

    const input = screen.getByPlaceholderText('Ask a question...')
    const searchButton = screen.getByRole('button', { name: /search/i })

    await userEvent.type(input, 'test')
    await userEvent.click(searchButton)

    await waitFor(() => {
      expect(screen.getByText('Test content')).toBeInTheDocument()
    })

    const clearButton = screen.getByRole('button', { name: /clear/i })
    await userEvent.click(clearButton)

    expect(screen.getByText('Welcome to RAG Application')).toBeInTheDocument()
    expect(screen.queryByText('Test content')).not.toBeInTheDocument()
  })

  it('toggles optimize query checkbox', async () => {
    render(<App />)

    const checkbox = screen.getByLabelText('Optimize Query')
    expect(checkbox).toBeChecked()

    await userEvent.click(checkbox)
    expect(checkbox).not.toBeChecked()
  })

  it('toggles rerank checkbox', async () => {
    render(<App />)

    const checkbox = screen.getByLabelText('Rerank Results')
    expect(checkbox).toBeChecked()

    await userEvent.click(checkbox)
    expect(checkbox).not.toBeChecked()
  })
})