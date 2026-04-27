import { useState } from 'react';
import './App.css';

function App() {
  const [text, setText] = useState('');
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState('');

  const analyze = async () => {
    if (!text.trim()) {
      setError('Please enter some text');
      return;
    }
    setLoading(true);
    setError('');
    setResult(null);

    try {
      const res = await fetch('/analyze', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text })
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || 'Failed to analyze');
      setResult(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="container">
      <h1>Sentiment Analyzer</h1>
      <p className="subtitle">Enter text to instantly detect its sentiment</p>
      
      <textarea 
        value={text}
        onChange={(e) => { setText(e.target.value); setError(''); }}
        placeholder="Type your statement here... (e.g. 'I absolutely love this new design!')"
        onKeyDown={(e) => e.key === 'Enter' && (e.ctrlKey || e.metaKey) && analyze()}
      />
      
      {error && <div style={{color: 'var(--neg)', marginBottom: '1rem', textAlign: 'center'}}>{error}</div>}

      <button onClick={analyze} disabled={loading || !text.trim()}>
        {loading ? <div className="spinner" /> : 'Analyze Sentiment'}
      </button>

      {result && (
        <div className="result">
          <div className={`badge ${result.sentiment.toLowerCase()}`}>
            {result.sentiment}
          </div>
          
          <div style={{textAlign: 'left'}}>
            <div className="flex-between">
              <span>Confidence Score</span>
              <span>{Math.round(result.confidence * 100)}%</span>
            </div>
            <div className="bar-bg">
              <div 
                className={`bar-fill ${result.sentiment.toLowerCase()}`}
                style={{width: `${Math.round(result.confidence * 100)}%`}}
              />
            </div>
          </div>
          
          <div className="flex-between" style={{marginTop: '1rem', justifyContent: 'center', gap: '1.5rem'}}>
            <span>Pos: {Math.round(result.probabilities.positive * 100)}%</span>
            <span>Neg: {Math.round(result.probabilities.negative * 100)}%</span>
          </div>
        </div>
      )}
    </div>
  );
}

export default App;
