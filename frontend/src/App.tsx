import React, { useState, useRef, useEffect } from 'react';
import './App.css';

interface Hotel {
  id: number;
  name: string;
  description: string;
  region?: string;
  country?: string;
  type?: string;
  atmosphere?: string;
  activities?: string[];
  services?: string[];
  best_months?: string[];
  score: number;
  description_score: number;
  structure_score: number;
}

interface ChatMessage {
  id: string;
  type: 'user' | 'bot';
  content: string;
  hotels?: Hotel[];
  searchType?: string;
  queryAnalysis?: any;
  timestamp: Date;
}

function App() {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [inputMessage, setInputMessage] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [isConfigured, setIsConfigured] = useState(false);
  const [connectionError, setConnectionError] = useState('');
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  useEffect(() => {
    // Check server status on app start
    checkHealthStatus();
  }, []);

  const checkHealthStatus = async () => {
    try {
      const response = await fetch('http://localhost:8001/api/health');
      const data = await response.json();
      
      setIsConfigured(data.qdrant_configured && data.openai_configured);
      setConnectionError('');
      
      if (data.status === 'not_configured') {
        setConnectionError('Server not configured. Please check your .env file and restart the server.');
        const errorMessage: ChatMessage = {
          id: Date.now().toString(),
          type: 'bot',
          content: '❌ Server configuration error. Please check your .env file contains valid QDRANT_URL, QDRANT_API_KEY, and OPENAI_API_KEY, then restart the server.',
          timestamp: new Date()
        };
        setMessages([errorMessage]);
      } else if (isConfigured && messages.length === 0) {
        // Add welcome message
        const welcomeMessage: ChatMessage = {
          id: Date.now().toString(),
          type: 'bot',
          content: '🏨 Welcome to the Structured RAG Hotel Search! I can help you find hotels using advanced search techniques. Try asking something like "I want a luxury resort in the Maldives with a spa and overwater bungalows."',
          timestamp: new Date()
        };
        setMessages([welcomeMessage]);
      }
    } catch (error) {
      console.error('Health check failed:', error);
      setConnectionError('Cannot connect to server. Please ensure the backend is running on port 8000.');
      const errorMessage: ChatMessage = {
        id: Date.now().toString(),
        type: 'bot',
        content: '❌ Cannot connect to the backend server. Please make sure you have started the backend with "python app.py".',
        timestamp: new Date()
      };
      setMessages([errorMessage]);
    }
  };

  const handleSendMessage = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!inputMessage.trim() || isLoading || !isConfigured) return;

    const userMessage: ChatMessage = {
      id: Date.now().toString(),
      type: 'user',
      content: inputMessage,
      timestamp: new Date()
    };

    setMessages(prev => [...prev, userMessage]);
    setInputMessage('');
    setIsLoading(true);

    try {
      const response = await fetch('http://localhost:8001/api/chat', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ message: inputMessage }),
      });

      if (response.ok) {
        const data = await response.json();
        
        const botMessage: ChatMessage = {
          id: (Date.now() + 1).toString(),
          type: 'bot',
          content: data.response,
          hotels: data.hotels,
          searchType: data.search_type,
          queryAnalysis: data.query_analysis,
          timestamp: new Date()
        };

        setMessages(prev => [...prev, botMessage]);
      } else {
        const errorData = await response.json();
        const errorMessage: ChatMessage = {
          id: (Date.now() + 1).toString(),
          type: 'bot',
          content: `Sorry, I encountered an error: ${errorData.detail}`,
          timestamp: new Date()
        };
        setMessages(prev => [...prev, errorMessage]);
      }
    } catch (error) {
      console.error('Chat failed:', error);
      const errorMessage: ChatMessage = {
        id: (Date.now() + 1).toString(),
        type: 'bot',
        content: 'Sorry, I encountered a network error. Please try again.',
        timestamp: new Date()
      };
      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  const renderHotel = (hotel: Hotel) => (
    <div key={hotel.id} className="hotel-card">
      <div className="hotel-header">
        <h4>{hotel.name}</h4>
        <div className="hotel-scores">
          <span className="combined-score">Score: {hotel.score}</span>
          <span className="detail-scores">
            (Desc: {hotel.description_score} | Struct: {hotel.structure_score})
          </span>
        </div>
      </div>
      
      <div className="hotel-details">
        {hotel.country && hotel.region && (
          <p><strong>Location:</strong> {hotel.country}, {hotel.region}</p>
        )}
        {hotel.type && <p><strong>Type:</strong> {hotel.type}</p>}
        {hotel.atmosphere && <p><strong>Atmosphere:</strong> {hotel.atmosphere}</p>}
        
        {hotel.activities && hotel.activities.length > 0 && (
          <p><strong>Activities:</strong> {hotel.activities.join(', ')}</p>
        )}
        
        {hotel.services && hotel.services.length > 0 && (
          <p><strong>Services:</strong> {hotel.services.join(', ')}</p>
        )}
        
        {hotel.best_months && hotel.best_months.length > 0 && (
          <p><strong>Best Months:</strong> {hotel.best_months.join(', ')}</p>
        )}
        
        <p className="hotel-description">{hotel.description}</p>
      </div>
    </div>
  );

  const renderMessage = (message: ChatMessage) => (
    <div key={message.id} className={`message ${message.type}`}> 
      <div className="message-content">
        {/* Only show hotels, search type, and query analysis for bot messages */}
        {message.type === 'bot' && (
          <>
            {message.searchType && (
              <div className="search-info">
                <span className={`search-type ${message.searchType}`}>
                  Search Type: {message.searchType === 'structured' ? 'Structured RAG' : 'Traditional RAG'}
                </span>
              </div>
            )}
            {message.queryAnalysis && (
              <div className="query-analysis">
                <details>
                  <summary>Query Analysis</summary>
                  <pre>{JSON.stringify(message.queryAnalysis, null, 2)}</pre>
                </details>
              </div>
            )}
            {message.hotels && message.hotels.length > 0 && (
              <div className="hotels-container">
                {message.hotels.map(renderHotel)}
              </div>
            )}
          </>
        )}
        {/* For user messages, show only the input text */}
        {message.type === 'user' && (
          <div className="message-text">{message.content}</div>
        )}
        <div className="message-timestamp">
          {message.timestamp.toLocaleTimeString()}
        </div>
      </div>
    </div>
  );

  return (
    <div className="App">
      <header className="App-header">
        <h1>🏨 Structured RAG Hotel Search</h1>
        <div className="header-controls">
          <div className={`status-indicator ${isConfigured ? 'configured' : 'not-configured'}`}>
            {isConfigured ? '✅ Ready' : '❌ Not Ready'}
          </div>
          {connectionError && (
            <button onClick={checkHealthStatus} className="retry-button">
              🔄 Retry Connection
            </button>
          )}
        </div>
      </header>

      <main className="chat-container">
        {connectionError && (
          <div className="error-banner">
            ⚠️ {connectionError}
          </div>
        )}
        
        <div className="messages-container">
          {messages.map(renderMessage)}
          {isLoading && (
            <div className="message bot">
              <div className="message-content">
                <div className="loading-indicator">Searching hotels...</div>
              </div>
            </div>
          )}
          <div ref={messagesEndRef} />
        </div>

        <form onSubmit={handleSendMessage} className="input-form">
          <input
            type="text"
            value={inputMessage}
            onChange={(e) => setInputMessage(e.target.value)}
            placeholder={isConfigured ? "Ask about hotels... (e.g., 'I want a luxury resort in Thailand with a spa')" : "Server not ready - please check configuration"}
            disabled={isLoading || !isConfigured}
            className="message-input"
          />
          <button 
            type="submit" 
            disabled={isLoading || !isConfigured || !inputMessage.trim()}
            className="send-button"
          >
            {isLoading ? '⏳' : '🔍'}
          </button>
        </form>
      </main>
    </div>
  );
}

export default App;
