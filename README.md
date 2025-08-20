# Smart Study Buddy Chat System

A simple AI-powered study assistant with LLM streaming, vector search, and real-time collaboration.

## Architecture

**Two-Server Setup:**
- **FastAPI Backend** (Port 8080): Handles AI, vector search, and data
- **Streamlit Frontend** (Port 8081): Provides user interface

```
Streamlit (8081) → HTTP requests → FastAPI (8080) → Gemini API + FAISS
```

## Features

### Core Functionality
- **Real-time LLM Streaming**: Live streaming responses from Gemini 2.0 Flash
- **Smart Search**: FAISS vector database for semantic search
- **Intelligent Flow**: Search materials first, then ask AI
- **Material Management**: Upload and organize study content
- **Visual Streaming**: See AI responses appear word-by-word in real-time

### Technical Features
- **Clean Architecture**: Separate FastAPI backend and Streamlit frontend
- **Server-Sent Events**: Proper streaming implementation
- **Simple Setup**: No complex database configuration
- **Vector Embeddings**: Semantic similarity search
- **REST API**: Well-defined API endpoints with streaming support

## Quick Start

1. **Install dependencies:**
```bash
pip install -r requirements.txt
```

2. **Configure API key:**
   - Edit `config.py`
   - Add your Gemini API key
   - Get key from: https://makersuite.google.com/app/apikey

3. **Run the system:**
```bash
python run.py
```

4. **Access the app:**
   - FastAPI Backend: http://localhost:8080
   - Streamlit Frontend: http://localhost:8081
   - Experience real-time streaming responses!

## Manual Setup

**Start Backend (Terminal 1):**
```bash
python fastapi_server.py
```

**Start Frontend (Terminal 2):**
```bash
streamlit run streamlit_app.py --server.port 8081
```

## Usage

1. **Add Study Materials:**
   - Use the sidebar form
   - Fill in title, subject, chapter, content
   - Materials are automatically indexed

2. **Ask Questions:**
   - Type questions in the chat
   - System searches materials first
   - AI provides context-aware answers

3. **View Results:**
   - See relevant materials highlighted
   - Get AI responses with context
   - Browse all uploaded materials

## API Endpoints

- `GET /` - Health check
- `POST /materials` - Add new material
- `GET /materials` - Get all materials
- `POST /search` - Search materials
- `POST /ask` - Ask AI question (simple response)
- `POST /ask-stream` - Ask AI question (streaming response)
- `GET /stats` - System statistics

## Testing

Run unit tests:
```bash
python unit_test.py
```

Tests cover:
- Gemini client initialization
- Study materials management
- Vector store operations
- Document indexing
- Basic functionality

## Project Structure

```
├── fastapi_server.py      # Backend API server with streaming
├── streamlit_app.py       # Basic frontend interface
├── streamlit_streaming.py # Optimized streaming frontend
├── gemini_client.py       # Gemini API integration
├── vector_store.py        # FAISS vector database
├── study_materials.py     # Materials management
├── config.py              # Configuration
├── unit_test.py           # Unit tests
├── run.py                 # Startup script
├── requirements.txt       # Dependencies
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

**Frontend connection error:**
- Ensure backend is running first
- Check FASTAPI_URL in config
- Try refreshing the page

**Search not working:**
- Upload some materials first
- Wait for indexing to complete
- Check vector store initialization

## Simple and Student-Friendly

- **No complex setup** - Just run and use
- **Clean interface** - Focus on learning
- **Fast responses** - Optimized for speed
- **Easy deployment** - Portable JSON storage