# Smart Study Buddy API System

A high-performance AI-powered study assistant API with real-time LLM streaming, vector search, and intelligent content management.

## Architecture

**Pure API System:**
- **FastAPI Backend** (Port 8080): Complete REST API with streaming support
- **Real-time Streaming**: Server-Sent Events (SSE) for live AI responses
- **Vector Search**: FAISS-powered semantic search through study materials

```
Client Applications → REST API (8080) → Gemini 2.0 Flash + FAISS Vector DB
```

## Features

### Core API Functionality
- **Real-time LLM Streaming**: Live streaming responses from Gemini 2.0 Flash via SSE
- **Semantic Search**: FAISS vector database for intelligent content discovery
- **Smart Context**: Automatically searches study materials before AI responses
- **Material Management**: Complete CRUD operations for study content
- **Vector Embeddings**: Automatic indexing of uploaded materials

### Technical Features
- **FastAPI Framework**: High-performance async API server
- **Server-Sent Events**: Real-time streaming without WebSockets
- **Vector Database**: FAISS for fast similarity search
- **JSON Storage**: Simple, portable data persistence
- **Auto-Indexing**: Automatic embedding generation and storage
- **Context-Aware AI**: AI responses enhanced with relevant study materials

## Quick Start

1. **Install Python 3.8+** (if not already installed)

2. **Clone this repository**

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

4. **Configure API key:**
   - Edit `config.py`
   - Add your Gemini API key
   - Get key from: https://makersuite.google.com/app/apikey

5. **Start the API server:**
```bash
python run.py
```
   Or directly:
```bash
python fastapi_server.py
```

6. **Access the API:**
   - API Server: http://localhost:8080
   - Interactive Docs: http://localhost:8080/docs
   - OpenAPI Spec: http://localhost:8080/openapi.json

## API Endpoints

### Core Endpoints
- `GET /` - Health check and API status
- `GET /stats` - System statistics (materials count, vector store size)

### Study Materials
- `POST /materials` - Upload new study material
- `GET /materials` - Retrieve all study materials

### AI & Search
- `POST /search` - Semantic search through study materials
- `POST /ask` - Get AI response (simple, non-streaming)
- `POST /ask-stream` - Get AI response (real-time streaming via SSE)

## API Usage Examples

### 1. Add Study Material
```bash
curl -X POST "http://localhost:8080/materials" \
  -H "Content-Type: application/json" \
  -d '{
    "title": "Python Programming Basics",
    "content": "Python is a high-level programming language known for its simplicity and readability. Variables are created by assignment, and Python uses indentation for code blocks.",
    "subject": "Programming",
    "chapter": "Chapter 1"
  }'
```

### 2. Search Study Materials
```bash
curl -X POST "http://localhost:8080/search" \
  -H "Content-Type: application/json" \
  -d '{"question": "What is Python programming?"}'
```

### 3. Ask AI (Simple Response)
```bash
curl -X POST "http://localhost:8080/ask" \
  -H "Content-Type: application/json" \
  -d '{"question": "Explain machine learning concepts"}'
```

### 4. Ask AI (Streaming Response)
```bash
curl -X POST "http://localhost:8080/ask-stream" \
  -H "Content-Type: application/json" \
  -d '{"question": "Explain artificial intelligence"}' \
  --no-buffer
```

### 5. Get System Statistics
```bash
curl -X GET "http://localhost:8080/stats"
```

## Response Formats

### Material Upload Response
```json
{
  "status": "success",
  "material": {
    "id": 1,
    "title": "Python Programming Basics",
    "content": "Python is a high-level...",
    "subject": "Programming",
    "chapter": "Chapter 1",
    "created_at": "2024-01-15T10:00:00"
  }
}
```

### Search Response
```json
[
  {
    "text": "Python is a high-level programming language...",
    "score": 0.85,
    "metadata": {
      "id": 1,
      "title": "Python Programming Basics",
      "subject": "Programming",
      "chapter": "Chapter 1"
    }
  }
]
```

### AI Response (Simple)
```json
{
  "response": "Machine learning is a subset of artificial intelligence...",
  "context_used": true,
  "context": "Python is a high-level programming language..."
}
```

### AI Response (Streaming SSE)
```
data: {"type": "context", "context_used": true, "context": "Python is..."}

data: {"type": "chunk", "content": "Machine "}

data: {"type": "chunk", "content": "learning "}

data: {"type": "chunk", "content": "is "}

data: {"type": "done"}
```

## Integration Examples

### Python Client
```python
import requests
import json

# Add material
response = requests.post(
    "http://localhost:8080/materials",
    json={
        "title": "AI Concepts",
        "content": "Artificial Intelligence involves...",
        "subject": "AI"
    }
)

# Stream AI response
response = requests.post(
    "http://localhost:8080/ask-stream",
    json={"question": "What is AI?"},
    stream=True
)

for line in response.iter_lines():
    if line.startswith(b'data: '):
        data = json.loads(line[6:])
        if data['type'] == 'chunk':
            print(data['content'], end='')
```

### JavaScript Client
```javascript
// Add material
fetch('http://localhost:8080/materials', {
  method: 'POST',
  headers: {'Content-Type': 'application/json'},
  body: JSON.stringify({
    title: 'Web Development',
    content: 'HTML, CSS, and JavaScript are...',
    subject: 'Web Dev'
  })
});

// Stream AI response
const eventSource = new EventSource('http://localhost:8080/ask-stream');
eventSource.onmessage = function(event) {
  const data = JSON.parse(event.data);
  if (data.type === 'chunk') {
    console.log(data.content);
  }
};
```

## Testing

### Integration Tests
Run comprehensive API integration tests:
```bash
python unit_test.py
```

**Tests include:**
- **Live API Testing** - Real server endpoint testing
- **Streaming Validation** - SSE format and content verification
- **Materials Management** - CRUD operations testing
- **Vector Search** - FAISS integration testing
- **Concurrent Requests** - Multiple simultaneous API calls
- **Error Handling** - Graceful failure management
- **End-to-End Flow** - Complete API workflow validation

## Project Structure

```
├── fastapi_server.py      # Main API server with streaming
├── gemini_client.py       # Gemini 2.0 Flash integration
├── vector_store.py        # FAISS vector database
├── study_materials.py     # Materials management system
├── config.py              # Configuration settings
├── unit_test.py           # Integration tests (8 comprehensive tests)
├── run.py                 # API server startup script
├── requirements.txt       # API dependencies
├── README.md              # This documentation
└── data/
    ├── study_materials.json    # JSON-based materials storage
    └── faiss_index/           # FAISS vector index files
```

## Configuration

Edit `config.py` to customize:
```python
# Gemini API
GEMINI_API_KEY = "your-api-key-here"
GEMINI_MODEL = "gemini-2.0-flash"

# Server
FASTAPI_PORT = 8080

# Vector Search
EMBEDDING_DIMENSION = 768
MAX_SEARCH_RESULTS = 5
SIMILARITY_THRESHOLD = 0.7
```

## Troubleshooting

### Common Issues

**API server won't start:**
- Check if port 8080 is available: `netstat -an | grep 8080`
- Verify Gemini API key in `config.py`
- Install missing dependencies: `pip install -r requirements.txt`

**Gemini API errors:**
- Verify API key is valid and active
- Check internet connection
- Monitor API quotas and limits

**Vector search not working:**
- Upload study materials first via `/materials` endpoint
- Wait for automatic indexing to complete
- Check FAISS index files in `data/faiss_index/`

**Streaming responses not working:**
- Verify client supports Server-Sent Events
- Check firewall settings for streaming connections
- Use `--no-buffer` flag with curl for immediate output

### Performance Tips
- Upload materials in smaller chunks for better indexing
- Use descriptive titles and include key concepts in content
- Monitor system stats via `/stats` endpoint
- Restart server periodically to clear memory

## Production Deployment

### Docker Deployment
```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8080
CMD ["python", "fastapi_server.py"]
```

### Environment Variables
```bash
export GEMINI_API_KEY="your-api-key"
export FASTAPI_PORT=8080
export FAISS_INDEX_PATH="/app/data/faiss_index"
```

## API Features Summary

✅ **High Performance** - FastAPI async framework
✅ **Real-time Streaming** - Server-Sent Events implementation
✅ **Smart Search** - FAISS vector similarity search
✅ **Context-Aware AI** - Study materials enhance AI responses
✅ **Simple Integration** - Standard REST API endpoints
✅ **Comprehensive Testing** - Real integration test suite
✅ **Easy Deployment** - Single server, JSON storage
✅ **Developer Friendly** - Interactive API documentation
✅ **Scalable Architecture** - Async processing, vector indexing