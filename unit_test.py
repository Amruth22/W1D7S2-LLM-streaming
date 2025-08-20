import pytest
import os
import json
import tempfile
import shutil
from unittest.mock import Mock, patch

from gemini_client import GeminiClient
from vector_store import VectorStore
from study_materials import StudyMaterialsManager

class TestGeminiClient:
    """Test Gemini client functionality"""
    
    def test_gemini_client_initialization(self):
        """Test 1: Gemini client can be initialized"""
        try:
            client = GeminiClient()
            assert client is not None
            assert hasattr(client, 'client')
            print("✓ Test 1 PASSED: Gemini client initialization")
        except Exception as e:
            print(f"✗ Test 1 FAILED: {e}")
            assert False

class TestStudyMaterialsManager:
    """Test study materials management"""
    
    def setup_method(self):
        """Setup test environment"""
        self.temp_dir = tempfile.mkdtemp()
        self.test_file = os.path.join(self.temp_dir, "test_materials.json")
        
        # Mock config
        with patch('study_materials.config') as mock_config:
            mock_config.STUDY_MATERIALS_FILE = self.test_file
            self.manager = StudyMaterialsManager()
    
    def teardown_method(self):
        """Cleanup test environment"""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_add_material(self):
        """Test 2: Adding study material"""
        try:
            material = self.manager.add_material(
                title="Test Title",
                content="Test Content",
                subject="Test Subject"
            )
            
            assert material is not None
            assert material['title'] == "Test Title"
            assert material['content'] == "Test Content"
            assert material['subject'] == "Test Subject"
            assert 'id' in material
            
            print("✓ Test 2 PASSED: Adding study material")
        except Exception as e:
            print(f"✗ Test 2 FAILED: {e}")
            assert False
    
    def test_get_materials(self):
        """Test 3: Getting all materials"""
        try:
            # Add test material
            self.manager.add_material("Title 1", "Content 1")
            self.manager.add_material("Title 2", "Content 2")
            
            materials = self.manager.get_all_materials()
            
            assert len(materials) == 2
            assert materials[0]['title'] == "Title 1"
            assert materials[1]['title'] == "Title 2"
            
            print("✓ Test 3 PASSED: Getting all materials")
        except Exception as e:
            print(f"✗ Test 3 FAILED: {e}")
            assert False

class TestVectorStore:
    """Test vector store functionality"""
    
    def setup_method(self):
        """Setup test environment"""
        self.temp_dir = tempfile.mkdtemp()
        
        # Mock config
        with patch('vector_store.config') as mock_config:
            mock_config.FAISS_INDEX_PATH = self.temp_dir
            mock_config.EMBEDDING_DIMENSION = 768
            self.vector_store = VectorStore()
    
    def teardown_method(self):
        """Cleanup test environment"""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_vector_store_initialization(self):
        """Test 4: Vector store initialization"""
        try:
            assert self.vector_store is not None
            assert self.vector_store.index is not None
            assert isinstance(self.vector_store.documents, list)
            
            stats = self.vector_store.get_stats()
            assert 'total_documents' in stats
            assert 'index_size' in stats
            
            print("✓ Test 4 PASSED: Vector store initialization")
        except Exception as e:
            print(f"✗ Test 4 FAILED: {e}")
            assert False
    
    def test_add_documents(self):
        """Test 5: Adding documents to vector store"""
        try:
            # Create mock embeddings
            import numpy as np
            texts = ["Test document 1", "Test document 2"]
            embeddings = [np.random.rand(768).tolist() for _ in texts]
            metadata = [{"id": 1, "title": "Doc 1"}, {"id": 2, "title": "Doc 2"}]
            
            # Add documents
            self.vector_store.add_documents(texts, embeddings, metadata)
            
            # Check stats
            stats = self.vector_store.get_stats()
            assert stats['total_documents'] == 2
            assert stats['index_size'] == 2
            
            print("✓ Test 5 PASSED: Adding documents to vector store")
        except Exception as e:
            print(f"✗ Test 5 FAILED: {e}")
            assert False

def run_tests():
    """Run all tests"""
    print("Running Smart Study Buddy Unit Tests")
    print("=" * 50)
    
    # Test 1: Gemini Client
    test_gemini = TestGeminiClient()
    test_gemini.test_gemini_client_initialization()
    
    # Test 2-3: Study Materials
    test_materials = TestStudyMaterialsManager()
    test_materials.setup_method()
    test_materials.test_add_material()
    test_materials.test_get_materials()
    test_materials.teardown_method()
    
    # Test 4-5: Vector Store
    test_vector = TestVectorStore()
    test_vector.setup_method()
    test_vector.test_vector_store_initialization()
    test_vector.test_add_documents()
    test_vector.teardown_method()
    
    print("=" * 50)
    print("All tests completed!")

if __name__ == "__main__":
    run_tests()