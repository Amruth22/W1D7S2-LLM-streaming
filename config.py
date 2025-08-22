# Configuration for Smart Study Buddy System

# Gemini API
GEMINI_API_KEY = ""
GEMINI_MODEL = "gemini-2.0-flash"
GEMINI_EMBEDDING_MODEL = "gemini-embedding-001"

# Server Ports
FASTAPI_PORT = 8080
STREAMLIT_PORT = 8081
FASTAPI_URL = f"http://localhost:{FASTAPI_PORT}"

# FAISS Settings
EMBEDDING_DIMENSION = 768
MAX_SEARCH_RESULTS = 5
SIMILARITY_THRESHOLD = 0.7

# File Paths
STUDY_MATERIALS_FILE = "data/study_materials.json"
FAISS_INDEX_PATH = "data/faiss_index"
