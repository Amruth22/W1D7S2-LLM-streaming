import os
import json
import numpy as np
import faiss
import config

class VectorStore:
    def __init__(self):
        self.index = None
        self.documents = []
        self.index_path = config.FAISS_INDEX_PATH
        self.documents_path = os.path.join(self.index_path, "documents.json")
        
        os.makedirs(self.index_path, exist_ok=True)
        self.load_index()
    
    def create_index(self):
        """Create new FAISS index"""
        self.index = faiss.IndexFlatIP(config.EMBEDDING_DIMENSION)
        self.documents = []
    
    def load_index(self):
        """Load existing index"""
        index_file = os.path.join(self.index_path, "faiss.index")
        
        if os.path.exists(index_file) and os.path.exists(self.documents_path):
            try:
                self.index = faiss.read_index(index_file)
                with open(self.documents_path, 'r') as f:
                    self.documents = json.load(f)
            except:
                self.create_index()
        else:
            self.create_index()
    
    def save_index(self):
        """Save index"""
        try:
            index_file = os.path.join(self.index_path, "faiss.index")
            faiss.write_index(self.index, index_file)
            
            with open(self.documents_path, 'w') as f:
                json.dump(self.documents, f, indent=2)
        except Exception as e:
            print(f"Save error: {e}")
    
    def add_documents(self, texts, embeddings, metadata=None):
        """Add documents to vector store"""
        try:
            embeddings_array = np.array(embeddings, dtype=np.float32)
            faiss.normalize_L2(embeddings_array)
            
            self.index.add(embeddings_array)
            
            for i, text in enumerate(texts):
                doc_metadata = metadata[i] if metadata and i < len(metadata) else {}
                self.documents.append({
                    'text': text,
                    'metadata': doc_metadata
                })
            
            self.save_index()
            
        except Exception as e:
            print(f"Add documents error: {e}")
    
    def search(self, query_embedding, k=5):
        """Search similar documents"""
        if not self.index or len(self.documents) == 0:
            return []
        
        try:
            query_array = np.array([query_embedding], dtype=np.float32)
            faiss.normalize_L2(query_array)
            
            scores, indices = self.index.search(query_array, min(k, len(self.documents)))
            
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx >= 0 and idx < len(self.documents):
                    doc = self.documents[idx]
                    results.append((doc['text'], float(score), doc['metadata']))
            
            return results
            
        except Exception as e:
            print(f"Search error: {e}")
            return []
    
    def get_stats(self):
        """Get statistics"""
        return {
            'total_documents': len(self.documents),
            'index_size': self.index.ntotal if self.index else 0
        }