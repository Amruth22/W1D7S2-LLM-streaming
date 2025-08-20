from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional
import uvicorn

from gemini_client import GeminiClient
from vector_store import VectorStore
from study_materials import StudyMaterialsManager
import config

# Initialize FastAPI app
app = FastAPI(title="Smart Study Buddy API", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize components
gemini_client = GeminiClient()
vector_store = VectorStore()
materials_manager = StudyMaterialsManager()

# Pydantic models
class MaterialCreate(BaseModel):
    title: str
    content: str
    subject: Optional[str] = ""
    chapter: Optional[str] = ""

class QuestionRequest(BaseModel):
    question: str

class SearchResponse(BaseModel):
    text: str
    score: float
    metadata: Dict

# Initialize vector store on startup
@app.on_event("startup")
async def startup_event():
    """Initialize vector store with existing materials"""
    try:
        materials = materials_manager.get_materials_for_embedding()
        if materials:
            texts = [m['text'] for m in materials]
            metadata = [m['metadata'] for m in materials]
            
            embeddings = gemini_client.get_embeddings(texts)
            if embeddings:
                vector_store.add_documents(texts, embeddings, metadata)
                print(f"Initialized with {len(materials)} materials")
    except Exception as e:
        print(f"Startup error: {e}")

# API Endpoints
@app.get("/")
async def root():
    """Health check"""
    return {"message": "Smart Study Buddy API is running"}

@app.post("/materials")
async def create_material(material: MaterialCreate):
    """Create new study material"""
    try:
        # Add material
        new_material = materials_manager.add_material(
            material.title, material.content, material.subject, material.chapter
        )
        
        # Add to vector store
        text = f"{material.title} {material.content}"
        embeddings = gemini_client.get_embeddings([text])
        
        if embeddings:
            vector_store.add_documents(
                texts=[text],
                embeddings=embeddings,
                metadata=[{
                    'id': new_material['id'],
                    'title': material.title,
                    'subject': material.subject,
                    'chapter': material.chapter
                }]
            )
        
        return {"status": "success", "material": new_material}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/materials")
async def get_materials():
    """Get all materials"""
    return materials_manager.get_all_materials()

@app.post("/search")
async def search_materials(request: QuestionRequest):
    """Search for relevant materials"""
    try:
        query_embeddings = gemini_client.get_embeddings([request.question])
        if not query_embeddings:
            return []
        
        results = vector_store.search(query_embeddings[0], k=config.MAX_SEARCH_RESULTS)
        
        # Filter by similarity threshold
        filtered_results = [
            SearchResponse(text=text, score=score, metadata=metadata)
            for text, score, metadata in results 
            if score >= config.SIMILARITY_THRESHOLD
        ]
        
        return filtered_results
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ask")
async def ask_question(request: QuestionRequest):
    """Get AI response to question"""
    try:
        # First search for relevant materials
        query_embeddings = gemini_client.get_embeddings([request.question])
        context = ""
        
        if query_embeddings:
            results = vector_store.search(query_embeddings[0], k=3)
            relevant_results = [
                text for text, score, metadata in results 
                if score >= config.SIMILARITY_THRESHOLD
            ]
            context = "\n".join(relevant_results[:2])
        
        # Get AI response
        response = gemini_client.simple_response(
            f"Context: {context}\n\nQuestion: {request.question}\n\nAnswer:"
        )
        
        return {
            "response": response,
            "context_used": bool(context),
            "context": context[:200] + "..." if len(context) > 200 else context
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/stats")
async def get_stats():
    """Get system statistics"""
    return {
        "materials": materials_manager.get_stats(),
        "vector_store": vector_store.get_stats()
    }

if __name__ == "__main__":
    uvicorn.run(
        "fastapi_server:app",
        host="localhost",
        port=config.FASTAPI_PORT,
        reload=True
    )