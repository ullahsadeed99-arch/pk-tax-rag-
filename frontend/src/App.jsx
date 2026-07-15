import { useRef, useState } from 'react'
import './App.css'

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:3001'

function SourceList({ sources }) {
  if (!sources || sources.length === 0) return null
  return (
    <ul className="sources">
      {sources.map((s, i) => (
        <li key={i}>
          <span className="source-badge">{i + 1}</span>
          {s.doc_name}
          {s.section ? `, Section ${s.section}` : ''}
          {s.section_title ? ` — ${s.section_title}` : ''} (p.{s.page_number})
        </li>
      ))}
    </ul>
  )
}

function App() {
  const [messages, setMessages] = useState([])
  const [input, setInput] = useState('')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const bottomRef = useRef(null)

  async function sendQuestion(e) {
    e.preventDefault()
    const question = input.trim()
    if (!question || loading) return

    setInput('')
    setError(null)
    setMessages((prev) => [...prev, { role: 'user', content: question }])
    setLoading(true)

    try {
      const res = await fetch(`${API_BASE_URL}/api/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question }),
      })
      const data = await res.json()
      if (!res.ok) throw new Error(data.error || 'Request failed')

      setMessages((prev) => [
        ...prev,
        { role: 'assistant', content: data.answer, sources: data.sources },
      ])
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
      setTimeout(() => bottomRef.current?.scrollIntoView({ behavior: 'smooth' }), 50)
    }
  }

  return (
    <div className="chat-app">
      <header className="chat-header">
        <h1>Pakistan Tax Law Assistant</h1>
        <p>Ask about the Income Tax Ordinance 2001, Income Tax Rules 2002, Sales Tax Act 1990, or Sales Tax Rules 2006.</p>
      </header>

      <main className="chat-window">
        {messages.length === 0 && (
          <div className="empty-state">
            Try: "What is the tax credit for charitable donations?" or "When must sales tax invoices be retained?"
          </div>
        )}
        {messages.map((m, i) => (
          <div key={i} className={`bubble ${m.role}`}>
            <div className="bubble-content">{m.content}</div>
            {m.role === 'assistant' && <SourceList sources={m.sources} />}
          </div>
        ))}
        {loading && <div className="bubble assistant pending">Thinking…</div>}
        {error && <div className="bubble error">Error: {error}</div>}
        <div ref={bottomRef} />
      </main>

      <form className="chat-input" onSubmit={sendQuestion}>
        <input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Ask a tax law question…"
          disabled={loading}
        />
        <button type="submit" disabled={loading || !input.trim()}>
          Send
        </button>
      </form>
    </div>
  )
}

export default App
