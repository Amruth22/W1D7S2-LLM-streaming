from google import genai
from google.genai import types
import config

class GeminiClient:
    def __init__(self):
        self.client = genai.Client(api_key=config.GEMINI_API_KEY)
    
    def stream_response(self, question, context=""):
        """Stream response from Gemini"""
        try:
            prompt = f"""You are a helpful study assistant. Answer clearly and simply.

Context: {context}

Question: {question}

Answer:"""

            contents = [types.Content(role="user", parts=[types.Part.from_text(text=prompt)])]
            
            for chunk in self.client.models.generate_content_stream(
                model=config.GEMINI_MODEL,
                contents=contents,
                config=types.GenerateContentConfig()
            ):
                if chunk.text:
                    yield chunk.text
                    
        except Exception as e:
            yield f"Error: {str(e)}"
    
    def get_embeddings(self, texts):
        """Get embeddings for texts"""
        try:
            if isinstance(texts, str):
                texts = [texts]
            
            result = self.client.models.embed_content(
                model=config.GEMINI_EMBEDDING_MODEL,
                contents=texts
            )
            
            return [embedding.values for embedding in result.embeddings]
            
        except Exception as e:
            print(f"Embedding error: {e}")
            return []
    
    def simple_response(self, question):
        """Get simple response"""
        try:
            contents = [types.Content(role="user", parts=[types.Part.from_text(text=question)])]
            
            response = self.client.models.generate_content(
                model=config.GEMINI_MODEL,
                contents=contents
            )
            
            return response.text
            
        except Exception as e:
            return f"Error: {str(e)}"