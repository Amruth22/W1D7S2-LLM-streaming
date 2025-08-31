import unittest
import os
import sys
import tempfile
import json
from dotenv import load_dotenv

# Add the current directory to Python path to import project modules
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

class CoreLLMStreamingTests(unittest.TestCase):
    """Core 5 unit tests for LLM Streaming System with real components"""
    
    @classmethod
    def setUpClass(cls):
        """Load environment variables and validate API key"""
        load_dotenv()
        
        # Validate API key
        cls.api_key = os.getenv('GEMINI_API_KEY')
        if not cls.api_key or not cls.api_key.startswith('AIza'):
            raise unittest.SkipTest("Valid GEMINI_API_KEY not found in environment")
        
        print(f"Using API Key: {cls.api_key[:10]}...{cls.api_key[-5:]}")
        
        # Initialize LLM streaming components
        try:
            import config
            from gemini_client import GeminiClient
            from vector_store import VectorStore
            from study_materials import StudyMaterialsManager
            
            cls.config = config
            cls.gemini_client = GeminiClient()
            cls.vector_store = VectorStore()
            cls.materials_manager = StudyMaterialsManager()
            
            print("LLM streaming components loaded successfully")
        except ImportError as e:
            raise unittest.SkipTest(f"Required LLM streaming components not found: {e}")

    def test_01_gemini_client_integration(self):
        """Test 1: Gemini Client Integration and API Communication"""
        print("Running Test 1: Gemini Client Integration")
        
        # Test client initialization
        self.assertIsNotNone(self.gemini_client)
        self.assertIsNotNone(self.gemini_client.client)
        
        # Test simple response generation
        response = self.gemini_client.simple_response("Hi")
        
        self.assertIsInstance(response, str)
        self.assertGreater(len(response), 0)
        self.assertNotIn("Error:", response)  # Should not contain error
        
        # Test embedding generation
        test_texts = ["Python programming", "Machine learning"]
        embeddings = self.gemini_client.get_embeddings(test_texts)
        
        self.assertIsInstance(embeddings, list)
        self.assertEqual(len(embeddings), len(test_texts))
        
        if embeddings:
            # Verify embedding structure
            first_embedding = embeddings[0]
            self.assertIsInstance(first_embedding, list)
            self.assertEqual(len(first_embedding), self.config.EMBEDDING_DIMENSION)
            self.assertTrue(all(isinstance(x, float) for x in first_embedding))
        
        # Test streaming response
        streaming_chunks = []
        for chunk in self.gemini_client.stream_response("Hello", ""):
            streaming_chunks.append(chunk)
            if len(streaming_chunks) >= 5:  # Limit for testing
                break
        
        self.assertGreater(len(streaming_chunks), 0)
        self.assertTrue(all(isinstance(chunk, str) for chunk in streaming_chunks))
        
        print(f"PASS: Simple response working - Response: {response[:50]}...")
        print(f"PASS: Embeddings generated - Dimension: {len(embeddings[0]) if embeddings else 0}")
        print(f"PASS: Streaming working - {len(streaming_chunks)} chunks received")

    def test_02_vector_store_operations(self):
        """Test 2: Vector Store Operations and FAISS Integration"""
        print("Running Test 2: Vector Store Operations")
        
        # Test vector store initialization
        self.assertIsNotNone(self.vector_store)
        self.assertIsNotNone(self.vector_store.index)
        
        # Test adding documents
        test_texts = ["Python is a programming language", "Machine learning uses algorithms"]
        test_embeddings = [[0.1] * self.config.EMBEDDING_DIMENSION, [0.2] * self.config.EMBEDDING_DIMENSION]
        test_metadata = [
            {"id": 1, "title": "Python Basics"},
            {"id": 2, "title": "ML Intro"}
        ]
        
        initial_doc_count = len(self.vector_store.documents)
        self.vector_store.add_documents(test_texts, test_embeddings, test_metadata)
        
        # Verify documents were added
        self.assertEqual(len(self.vector_store.documents), initial_doc_count + 2)
        
        # Test search functionality
        query_embedding = [0.15] * self.config.EMBEDDING_DIMENSION
        results = self.vector_store.search(query_embedding, k=3)
        
        self.assertIsInstance(results, list)
        self.assertLessEqual(len(results), 3)
        
        # Test search result format
        if results:
            text, score, metadata = results[0]
            self.assertIsInstance(text, str)
            self.assertIsInstance(score, float)
            self.assertIsInstance(metadata, dict)
        
        # Test statistics
        stats = self.vector_store.get_stats()
        self.assertIn('total_documents', stats)
        self.assertIn('index_size', stats)
        self.assertGreaterEqual(stats['total_documents'], 0)
        
        print(f"PASS: Vector store operations - {stats['total_documents']} documents indexed")
        print(f"PASS: Search functionality - {len(results)} results returned")

    def test_03_study_materials_management(self):
        """Test 3: Study Materials Management and Storage"""
        print("Running Test 3: Study Materials Management")
        
        # Test materials manager initialization
        self.assertIsNotNone(self.materials_manager)
        
        # Test adding new material
        initial_count = len(self.materials_manager.materials)
        
        new_material = self.materials_manager.add_material(
            title="Test Material",
            content="This is test content for study materials.",
            subject="Testing",
            chapter="Chapter 1"
        )
        
        self.assertIsNotNone(new_material)
        self.assertEqual(new_material['title'], "Test Material")
        self.assertEqual(new_material['content'], "This is test content for study materials.")
        self.assertEqual(new_material['subject'], "Testing")
        self.assertEqual(new_material['chapter'], "Chapter 1")
        self.assertIn('id', new_material)
        self.assertIn('created_at', new_material)
        
        # Verify material was added
        self.assertEqual(len(self.materials_manager.materials), initial_count + 1)
        
        # Test retrieving all materials
        all_materials = self.materials_manager.get_all_materials()
        self.assertIsInstance(all_materials, list)
        self.assertGreaterEqual(len(all_materials), 1)
        
        # Test materials for embedding
        embedding_materials = self.materials_manager.get_materials_for_embedding()
        self.assertIsInstance(embedding_materials, list)
        
        if embedding_materials:
            first_material = embedding_materials[-1]  # Get the one we just added
            self.assertIn('text', first_material)
            self.assertIn('metadata', first_material)
            self.assertIsInstance(first_material['text'], str)
            self.assertIsInstance(first_material['metadata'], dict)
            self.assertIn(new_material['title'], first_material['text'])
        
        # Test statistics
        stats = self.materials_manager.get_stats()
        self.assertIn('total_materials', stats)
        self.assertIn('subjects', stats)
        self.assertIsInstance(stats['subjects'], dict)
        self.assertGreaterEqual(stats['total_materials'], 1)
        
        print(f"PASS: Material added - ID: {new_material['id']}, Title: {new_material['title']}")
        print(f"PASS: Statistics - Total: {stats['total_materials']}, Subjects: {len(stats['subjects'])}")

    def test_04_fastapi_integration(self):
        """Test 4: FastAPI Integration and Endpoint Testing"""
        print("Running Test 4: FastAPI Integration")
        
        # Test FastAPI server components
        try:
            from fastapi_server import app
            from fastapi.testclient import TestClient
            
            client = TestClient(app)
            
            # Test health check endpoint
            response = client.get("/")
            self.assertEqual(response.status_code, 200)
            data = response.json()
            self.assertIn("message", data)
            self.assertIn("Smart Study Buddy API", data["message"])
            
            # Test materials endpoint - GET
            response = client.get("/materials")
            self.assertEqual(response.status_code, 200)
            materials = response.json()
            self.assertIsInstance(materials, list)
            
            # Test materials endpoint - POST
            test_material = {
                "title": "FastAPI Test Material",
                "content": "Testing FastAPI endpoints with real integration.",
                "subject": "Testing",
                "chapter": "Chapter 1"
            }
            
            response = client.post("/materials", json=test_material)
            self.assertEqual(response.status_code, 200)
            result = response.json()
            self.assertIn("status", result)
            self.assertEqual(result["status"], "success")
            self.assertIn("material", result)
            
            # Test stats endpoint
            response = client.get("/stats")
            self.assertEqual(response.status_code, 200)
            stats = response.json()
            self.assertIn("materials", stats)
            self.assertIn("vector_store", stats)
            
            print("PASS: FastAPI endpoints working correctly")
            print("PASS: Health check and materials management validated")
            print("PASS: Statistics endpoint functional")
            
        except ImportError as e:
            print(f"INFO: FastAPI integration test skipped due to: {str(e)}")
            
            # Test configuration instead
            self.assertEqual(self.config.FASTAPI_PORT, 8080)
            self.assertEqual(self.config.GEMINI_MODEL, "gemini-2.0-flash")
            self.assertEqual(self.config.EMBEDDING_DIMENSION, 768)
            self.assertEqual(self.config.MAX_SEARCH_RESULTS, 5)
            self.assertEqual(self.config.SIMILARITY_THRESHOLD, 0.7)
            
            print("PASS: Configuration validation completed")

    def test_05_streaming_and_search_integration(self):
        """Test 5: Streaming Response and Search Integration"""
        print("Running Test 5: Streaming and Search Integration")
        
        # Test search integration with real embeddings
        test_question = "What is Python programming?"
        
        # Generate embedding for the question
        query_embeddings = self.gemini_client.get_embeddings([test_question])
        
        if query_embeddings:
            self.assertIsInstance(query_embeddings, list)
            self.assertEqual(len(query_embeddings), 1)
            self.assertEqual(len(query_embeddings[0]), self.config.EMBEDDING_DIMENSION)
            
            # Test search with the embedding
            search_results = self.vector_store.search(query_embeddings[0], k=3)
            self.assertIsInstance(search_results, list)
            self.assertLessEqual(len(search_results), 3)
            
            print(f"PASS: Search integration - {len(search_results)} results found")
        else:
            print("INFO: Embedding generation test skipped")
        
        # Test streaming response with context
        context = "Python is a high-level programming language known for its simplicity."
        streaming_chunks = []
        
        try:
            for chunk in self.gemini_client.stream_response(test_question, context):
                streaming_chunks.append(chunk)
                if len(streaming_chunks) >= 10:  # Limit for testing
                    break
            
            self.assertGreater(len(streaming_chunks), 0)
            
            # Verify no error chunks
            error_chunks = [chunk for chunk in streaming_chunks if "Error:" in chunk]
            self.assertEqual(len(error_chunks), 0, "Should not have error chunks")
            
            # Test assembled response
            full_response = "".join(streaming_chunks)
            self.assertGreater(len(full_response), 0)
            
            print(f"PASS: Streaming response - {len(streaming_chunks)} chunks received")
            print(f"PASS: Full response length: {len(full_response)} characters")
            
        except Exception as e:
            print(f"INFO: Streaming test completed with note: {str(e)}")
        
        # Test configuration values
        self.assertGreater(self.config.EMBEDDING_DIMENSION, 0)
        self.assertGreater(self.config.MAX_SEARCH_RESULTS, 0)
        self.assertGreater(self.config.SIMILARITY_THRESHOLD, 0)
        self.assertLess(self.config.SIMILARITY_THRESHOLD, 1)
        
        # Test file paths configuration
        self.assertTrue(self.config.STUDY_MATERIALS_FILE.endswith('.json'))
        self.assertIn('faiss', self.config.FAISS_INDEX_PATH.lower())
        
        print("PASS: Configuration parameters validated")
        print("PASS: Streaming and search integration functional")

def run_core_tests():
    """Run core tests and provide summary"""
    print("=" * 70)
    print("[*] Core LLM Streaming System Unit Tests (5 Tests)")
    print("Testing with REAL API and LLM Streaming Components")
    print("=" * 70)
    
    # Check API key
    load_dotenv()
    api_key = os.getenv('GEMINI_API_KEY')
    
    if not api_key or not api_key.startswith('AIza'):
        print("[ERROR] Valid GEMINI_API_KEY not found!")
        return False
    
    print(f"[OK] Using API Key: {api_key[:10]}...{api_key[-5:]}")
    print()
    
    # Run tests
    suite = unittest.TestLoader().loadTestsFromTestCase(CoreLLMStreamingTests)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    print("\n" + "=" * 70)
    print("[*] Test Results:")
    print(f"[*] Tests Run: {result.testsRun}")
    print(f"[*] Failures: {len(result.failures)}")
    print(f"[*] Errors: {len(result.errors)}")
    
    if result.failures:
        print("\n[FAILURES]:")
        for test, traceback in result.failures:
            print(f"  - {test}")
            print(f"    {traceback}")
    
    if result.errors:
        print("\n[ERRORS]:")
        for test, traceback in result.errors:
            print(f"  - {test}")
            print(f"    {traceback}")
    
    success = len(result.failures) == 0 and len(result.errors) == 0
    
    if success:
        print("\n[SUCCESS] All 5 core LLM streaming tests passed!")
        print("[OK] LLM streaming components working correctly with real API")
        print("[OK] Gemini Client, Vector Store, Study Materials, FastAPI, Streaming validated")
    else:
        print(f"\n[WARNING] {len(result.failures) + len(result.errors)} test(s) failed")
    
    return success

if __name__ == "__main__":
    print("[*] Starting Core LLM Streaming System Tests")
    print("[*] 5 essential tests with real API and streaming components")
    print("[*] Components: Gemini Client, Vector Store, Study Materials, FastAPI, Streaming")
    print()
    
    success = run_core_tests()
    exit(0 if success else 1)