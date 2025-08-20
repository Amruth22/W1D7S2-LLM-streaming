# Smart Study Buddy Chat System

A simple AI-powered study assistant with LLM streaming, vector search, and real-time collaboration.

## Architecture

**API-Only System:**
- **FastAPI Backend** (Port 8080): Handles AI, vector search, and data
- **REST API Endpoints**: Complete API for external clients

```
Client Applications → HTTP requests → FastAPI (8080) → Gemini API + FAISS
```

## Features

### Core Functionality
- **Real-time LLM Streaming**: Live streaming responses from Gemini 2.0 Flash
- **Smart Search**: FAISS vector database for semantic search
- **Intelligent Flow**: Search materials first, then ask AI
- **Material Management**: Upload and organize study content
- **Server-Sent Events**: Real-time streaming via SSE

### Technical Features
- **FastAPI Backend**: High-performance API server
- **Server-Sent Events**: Proper streaming implementation
- **Simple Setup**: No complex database configuration
- **Vector Embeddings**: Semantic similarity search
- **REST API**: Well-defined API endpoints with streaming support
- **Real Integration Tests**: Live API testing with actual server

## Quick Start

1. **Install dependencies:**
```bash
pip install -r requirements.txt
```

2. **Configure API key:**
   - Edit `config.py`
   - Add your Gemini API key
   - Get key from: https://makersuite.google.com/app/apikey

3. **Run the API server:**
```bash
python run.py
```
   Or directly:
```bash
python fastapi_server.py
```

4. **Access the API:**
   - API Server: http://localhost:8080
   - API Documentation: http://localhost:8080/docs
   - Experience real-time streaming via API endpoints!

## Manual Setup

**Start API Server:**
```bash
python fastapi_server.py
```

**Access API Documentation:**
- Interactive docs: http://localhost:8080/docs
- OpenAPI spec: http://localhost:8080/openapi.json

## API Usage

1. **Add Study Materials:**
```bash
curl -X POST "http://localhost:8080/materials" \
  -H "Content-Type: application/json" \
  -d '{"title":"Python Basics","content":"Python is...","subject":"Programming"}'
```

2. **Search Materials:**
```bash
curl -X POST "http://localhost:8080/search" \
  -H "Content-Type: application/json" \
  -d '{"question":"What is Python?"}'
```

3. **Ask AI (Simple):**
```bash
curl -X POST "http://localhost:8080/ask" \
  -H "Content-Type: application/json" \
  -d '{"question":"Explain machine learning"}'
```

4. **Ask AI (Streaming):**
```bash
curl -X POST "http://localhost:8080/ask-stream" \
  -H "Content-Type: application/json" \
  -d '{"question":"Explain AI"}' \
  --no-buffer
```

## API Endpoints

- `GET /` - Health check
- `POST /materials` - Add new material
- `GET /materials` - Get all materials
- `POST /search` - Search materials
- `POST /ask` - Ask AI question (simple response)
- `POST /ask-stream` - Ask AI question (streaming response)
- `GET /stats` - System statistics

## Testing

### Integration Tests
Run real API integration tests:
```bash
python unit_test.py
```

Tests cover:
- **Real API endpoint testing** - Live server testing
- **Live streaming responses** - Actual SSE streaming
- **Materials management** - CRUD operations
- **Vector search** - FAISS integration
- **Concurrent streaming** - Multiple simultaneous requests
- **Error handling** - Graceful failure management
- **Response validation** - Format and content verification
- **End-to-end flow** - Complete API workflow

## Project Structure

```
├── fastapi_server.py      # Main API server with streaming
├── gemini_client.py       # Gemini API integration
├── vector_store.py        # FAISS vector database
├── study_materials.py     # Materials management
├── config.py              # Configuration
├── unit_test.py           # Real integration tests (8 tests)
├── run.py                 # API startup script
├── requirements.txt       # API dependencies
├── README.md              # Documentation
└── data/
    ├── study_materials.json    # Materials storage
    └── faiss_index/           # Vector index files
```

## Configuration

Edit `config.py`:
```python
GEMINI_API_KEY = "your-api-key-here"
FASTAPI_PORT = 8080
STREAMLIT_PORT = 8081
```

## Troubleshooting

**Backend not starting:**
- Check if port 8080 is available
- Verify Gemini API key
- Install missing dependencies

**API connection error:**
- Ensure API server is running on port 8080
- Check if port 8080 is available
- Verify API endpoints at http://localhost:8080/docs

**Search not working:**
- Upload some materials first
- Wait for indexing to complete
- Check vector store initialization

## Simple and Developer-Friendly

- **No complex setup** - Just run and use
- **Clean API** - Well-documented endpoints
- **Fast responses** - Optimized for speed
- **Easy integration** - Standard REST API
- **Real-time streaming** - Server-Sent Events
- **Portable storage** - JSON-based data