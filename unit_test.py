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
            return True
        except Exception as e:
            print(f"✗ Test 1 FAILED: {e}")
            return False

def test_add_material():
    """Test 2: Adding study material"""
    try:
        # Create isolated test environment
        temp_dir = tempfile.mkdtemp()
        test_file = os.path.join(temp_dir, "test_materials.json")
        
        # Mock config for this test
        with patch('study_materials.config') as mock_config:
            mock_config.STUDY_MATERIALS_FILE = test_file
            manager = StudyMaterialsManager()
            
            material = manager.add_material(
                title="Test Title",
                content="Test Content",
                subject="Test Subject"
            )
            
            assert material is not None
            assert material['title'] == "Test Title"
            assert material['content'] == "Test Content"
            assert material['subject'] == "Test Subject"
            assert 'id' in material
        
        # Cleanup
        shutil.rmtree(temp_dir)
        print("✓ Test 2 PASSED: Adding study material")
        
    except Exception as e:
        print(f"✗ Test 2 FAILED: {e}")
        return False
    return True

def test_get_materials():
    """Test 3: Getting all materials"""
    try:
        # Create fresh isolated test environment
        temp_dir = tempfile.mkdtemp()
        test_file = os.path.join(temp_dir, "test_materials.json")
        
        # Mock config for this test
        with patch('study_materials.config') as mock_config:
            mock_config.STUDY_MATERIALS_FILE = test_file
            manager = StudyMaterialsManager()
            
            # Start with empty materials
            initial_materials = manager.get_all_materials()
            assert len(initial_materials) == 0, f"Expected 0 materials, got {len(initial_materials)}"
            
            # Add test materials
            manager.add_material("Title 1", "Content 1")
            manager.add_material("Title 2", "Content 2")
            
            # Check materials
            materials = manager.get_all_materials()
            assert len(materials) == 2, f"Expected 2 materials, got {len(materials)}"
            assert materials[0]['title'] == "Title 1"
            assert materials[1]['title'] == "Title 2"
        
        # Cleanup
        shutil.rmtree(temp_dir)
        print("✓ Test 3 PASSED: Getting all materials")
        
    except Exception as e:
        print(f"✗ Test 3 FAILED: {e}")
        return False
    return True

def test_vector_store_initialization():
    """Test 4: Vector store initialization"""
    try:
        # Create isolated test environment
        temp_dir = tempfile.mkdtemp()
        
        # Mock config for this test
        with patch('vector_store.config') as mock_config:
            mock_config.FAISS_INDEX_PATH = temp_dir
            mock_config.EMBEDDING_DIMENSION = 768
            vector_store = VectorStore()
            
            assert vector_store is not None
            assert vector_store.index is not None
            assert isinstance(vector_store.documents, list)
            
            stats = vector_store.get_stats()
            assert 'total_documents' in stats
            assert 'index_size' in stats
        
        # Cleanup
        shutil.rmtree(temp_dir)
        print("✓ Test 4 PASSED: Vector store initialization")
        
    except Exception as e:
        print(f"✗ Test 4 FAILED: {e}")
        return False
    return True

def test_add_documents():
    """Test 5: Adding documents to vector store"""
    try:
        # Create isolated test environment
        temp_dir = tempfile.mkdtemp()
        
        # Mock config for this test
        with patch('vector_store.config') as mock_config:
            mock_config.FAISS_INDEX_PATH = temp_dir
            mock_config.EMBEDDING_DIMENSION = 768
            vector_store = VectorStore()
            
            # Create mock embeddings
            import numpy as np
            texts = ["Test document 1", "Test document 2"]
            embeddings = [np.random.rand(768).tolist() for _ in texts]
            metadata = [{"id": 1, "title": "Doc 1"}, {"id": 2, "title": "Doc 2"}]
            
            # Add documents
            vector_store.add_documents(texts, embeddings, metadata)
            
            # Check stats
            stats = vector_store.get_stats()
            assert stats['total_documents'] == 2
            assert stats['index_size'] == 2
        
        # Cleanup
        shutil.rmtree(temp_dir)
        print("✓ Test 5 PASSED: Adding documents to vector store")
        
    except Exception as e:
        print(f"✗ Test 5 FAILED: {e}")
        return False
    return True

def run_tests():
    """Run all tests"""
    print("Running Smart Study Buddy Unit Tests")
    print("=" * 50)
    
    tests = [
        ("Gemini Client Initialization", lambda: TestGeminiClient().test_gemini_client_initialization()),
        ("Adding Study Material", test_add_material),
        ("Getting All Materials", test_get_materials),
        ("Vector Store Initialization", test_vector_store_initialization),
        ("Adding Documents to Vector Store", test_add_documents)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            if result is not False:  # None or True means passed
                passed += 1
        except Exception as e:
            print(f"✗ {test_name} ERROR: {e}")
    
    print("=" * 50)
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("✅ All tests passed!")
        return True
    else:
        print("❌ Some tests failed.")
        return False

if __name__ == "__main__":
    run_tests()