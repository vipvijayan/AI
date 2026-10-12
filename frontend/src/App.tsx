import { useState } from 'react'
import { queryRag, getStats, type QueryResponse, type StatsResponse, type QueryParams } from './api/client'
import './App.css'

function App() {
  const [query, setQuery] = useState('')
  const [results, setResults] = useState<QueryResponse | null>(null)
  const [stats, setStats] = useState<StatsResponse | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const [k, setK] = useState(10)
  const [optimizeQuery, setOptimizeQuery] = useState(true)
  const [rerank, setRerank] = useState(true)
  const [rerankK, setRerankK] = useState(5)

  const handleSearch = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!query.trim()) return

    setLoading(true)
    setError(null)

    try {
      const params: QueryParams = {
        query: query.trim(),
        k,
        optimize_query: optimizeQuery,
        rerank,
        rerank_k: rerankK,
      }

      const response = await queryRag(params)
      setResults(response)
      await loadStats()
    } catch (err) {
      setError('Failed to execute query. Please try again.')
      console.error(err)
    } finally {
      setLoading(false)
    }
  }

  const loadStats = async () => {
    try {
      const statsData = await getStats()
      setStats(statsData)
    } catch {
      // Ignore stats errors
    }
  }

  const handleClear = () => {
    setQuery('')
    setResults(null)
    setError(null)
  }

  return (
    <div className="container">
      <header>
        <h1>RAG Application</h1>
        <p style={{ opacity: 0.8, marginTop: 4 }}>Hybrid Retrieval with Cross-Encoder Reranking</p>
      </header>

      {stats && (
        <div className="stats">
          <div className="stats-grid">
            <div className="stat-item">
              <div className="stat-value">{stats.documents_indexed}</div>
              <div className="stat-label">Documents</div>
            </div>
            <div className="stat-item">
              <div className="stat-value">{stats.queries_processed}</div>
              <div className="stat-label">Queries</div>
            </div>
            <div className="stat-item">
              <div className="stat-value">{stats.retriever}</div>
              <div className="stat-label">Retriever</div>
            </div>
          </div>
        </div>
      )}

      <form onSubmit={handleSearch} className="search-form">
        <input
          type="text"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="Ask a question..."
          className="search-input"
          disabled={loading}
        />
        <button type="submit" className="search-button" disabled={loading || !query.trim()}>
          {loading ? 'Searching...' : 'Search'}
        </button>
      </form>

      <div className="options">
        <div className="option">
          <input type="checkbox" id="opt-k" checked={true} disabled />
          <label htmlFor="opt-k">Top K: {k}</label>
        </div>
        <div className="option">
          <input
            type="checkbox"
            id="opt-optimize"
            checked={optimizeQuery}
            onChange={(e) => setOptimizeQuery(e.target.checked)}
          />
          <label htmlFor="opt-optimize">Optimize Query</label>
        </div>
        <div className="option">
          <input
            type="checkbox"
            id="opt-rerank"
            checked={rerank}
            onChange={(e) => setRerank(e.target.checked)}
          />
          <label htmlFor="opt-rerank">Rerank Results</label>
        </div>
      </div>

      {error && <div className="error">{error}</div>}

      {results ? (
        results.total_results > 0 ? (
          <div className="results">
            {results.results.map((result, index) => (
              <div key={result.document_id} className="result-card">
                <div className="result-header">
                  <span className="result-rank">#{index + 1}</span>
                  <span className="result-score">Score: {result.score.toFixed(3)}</span>
                </div>
                <div className="result-content">{result.content}</div>
                <div className="result-meta">
                  ID: {result.document_id}
                  {result.metadata.title && ` | Title: ${result.metadata.title}`}
                </div>
              </div>
            ))}
          </div>
        ) : (
          <div className="empty-state">
            <h2>No Results Found</h2>
            <p>Try a different query or adjust search parameters</p>
          </div>
        )
      ) : (
        <div className="empty-state">
          <h2>Welcome to RAG Application</h2>
          <p>Enter a question above to search the knowledge base</p>
        </div>
      )}

      <button onClick={handleClear} className="search-button" style={{ marginTop: 20, background: '#95a5a6' }}>
        Clear
      </button>
    </div>
  )
}

export default App