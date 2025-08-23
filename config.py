# Configuration for Smart Study Buddy System

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Gemini API - loaded from environment variables
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
GEMINI_EMBEDDING_MODEL = os.getenv("GEMINI_EMBEDDING_MODEL", "gemini-embedding-001")

# Validate API key
if not GEMINI_API_KEY:
    print("⚠️  WARNING: GEMINI_API_KEY not found in .env file!")
    print("Please add your Gemini API key to the .env file:")
    print("1. Get your API key from: https://makersuite.google.com/app/apikey")
    print("2. Add 'GEMINI_API_KEY=your-api-key-here' to .env file")
else:
    print(f"✅ Gemini API key loaded successfully (length: {len(GEMINI_API_KEY)})")

# Server Ports
FASTAPI_PORT = 8080
STREAMLIT_PORT = 8081
FASTAPI_URL = f"http://localhost:{FASTAPI_PORT}"

# FAISS Settings - can be overridden via environment variables
EMBEDDING_DIMENSION = int(os.getenv("EMBEDDING_DIMENSION", 768))
MAX_SEARCH_RESULTS = int(os.getenv("MAX_SEARCH_RESULTS", 5))
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", 0.7))

# File Paths - can be overridden via environment variables
STUDY_MATERIALS_FILE = os.getenv("STUDY_MATERIALS_FILE", "data/study_materials.json")
FAISS_INDEX_PATH = os.getenv("FAISS_INDEX_PATH", "data/faiss_index")
