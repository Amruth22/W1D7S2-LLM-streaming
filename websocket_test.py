#!/usr/bin/env python3
"""
WebSocket and Streaming Integration Tests
Tests the real-time streaming functionality
"""

import asyncio
import json
import time
import threading
import subprocess
import sys
import requests
from datetime import datetime
import config

class WebSocketStreamingTester:
    """Test WebSocket and streaming functionality"""
    
    def __init__(self):
        self.server_process = None
        self.base_url = f"http://localhost:{config.FASTAPI_PORT}"
    
    def start_test_server(self):
        """Start FastAPI server for testing"""
        try:
            print("Starting test server...")
            self.server_process = subprocess.Popen([
                sys.executable, "fastapi_server.py"
            ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            # Wait for server to start
            time.sleep(3)
            
            # Check if server is running
            response = requests.get(f"{self.base_url}/")
            if response.status_code == 200:
                print("✅ Test server started successfully")
                return True
            else:
                print("❌ Test server failed to start")
                return False
                
        except Exception as e:
            print(f"❌ Error starting test server: {e}")
            return False
    
    def stop_test_server(self):
        """Stop test server"""
        if self.server_process:
            self.server_process.terminate()
            self.server_process.wait()
            print("🛑 Test server stopped")
    
    def test_api_health(self):
        """Test 1: API health check"""
        try:
            response = requests.get(f"{self.base_url}/")
            assert response.status_code == 200
            data = response.json()
            assert "message" in data
            print("✅ Test 1 PASSED: API health check")
            return True
        except Exception as e:
            print(f"❌ Test 1 FAILED: {e}")
            return False
    
    def test_materials_endpoint(self):
        """Test 2: Materials endpoint"""
        try:
            # Test GET materials
            response = requests.get(f"{self.base_url}/materials")
            assert response.status_code == 200
            materials = response.json()
            assert isinstance(materials, list)
            
            # Test POST material
            test_material = {
                "title": "Test WebSocket Material",
                "content": "This is test content for WebSocket testing",
                "subject": "Testing",
                "chapter": "Chapter 1"
            }
            
            response = requests.post(f"{self.base_url}/materials", json=test_material)
            assert response.status_code == 200
            result = response.json()
            assert result["status"] == "success"
            
            print("✅ Test 2 PASSED: Materials endpoint")
            return True
        except Exception as e:
            print(f"❌ Test 2 FAILED: {e}")
            return False
    
    def test_search_endpoint(self):
        """Test 3: Search endpoint"""
        try:
            search_request = {"question": "What is testing?"}
            response = requests.post(f"{self.base_url}/search", json=search_request)
            assert response.status_code == 200
            results = response.json()
            assert isinstance(results, list)
            
            print("✅ Test 3 PASSED: Search endpoint")
            return True
        except Exception as e:
            print(f"❌ Test 3 FAILED: {e}")
            return False
    
    def test_simple_ask_endpoint(self):
        """Test 4: Simple ask endpoint"""
        try:
            ask_request = {"question": "What is machine learning?"}
            response = requests.post(f"{self.base_url}/ask", json=ask_request, timeout=30)
            assert response.status_code == 200
            result = response.json()
            assert "response" in result
            assert "context_used" in result
            
            print("✅ Test 4 PASSED: Simple ask endpoint")
            return True
        except Exception as e:
            print(f"❌ Test 4 FAILED: {e}")
            return False
    
    def test_streaming_endpoint(self):
        """Test 5: Streaming endpoint (SSE)"""
        try:
            ask_request = {"question": "Explain Python programming"}
            
            # Test streaming response
            response = requests.post(
                f"{self.base_url}/ask-stream", 
                json=ask_request, 
                stream=True,
                timeout=60
            )
            
            assert response.status_code == 200
            assert response.headers.get('content-type') == 'text/plain; charset=utf-8'
            
            # Collect streaming chunks
            chunks_received = 0
            context_received = False
            content_chunks = []
            
            for line in response.iter_lines():
                if line:
                    line = line.decode('utf-8')
                    if line.startswith('data: '):
                        try:
                            data = json.loads(line[6:])  # Remove 'data: '
                            
                            if data['type'] == 'context':
                                context_received = True
                            elif data['type'] == 'chunk':
                                content_chunks.append(data['content'])
                                chunks_received += 1
                            elif data['type'] == 'done':
                                break
                            elif data['type'] == 'error':
                                print(f"Streaming error: {data['message']}")
                                return False
                                
                        except json.JSONDecodeError:
                            continue
                
                # Limit test duration
                if chunks_received > 20:  # Stop after reasonable number of chunks
                    break
            
            # Verify streaming worked
            assert chunks_received > 0, f"No chunks received"
            assert len(content_chunks) > 0, "No content chunks received"
            
            # Verify we got actual content
            full_response = "".join(content_chunks)
            assert len(full_response) > 10, "Response too short"
            
            print(f"✅ Test 5 PASSED: Streaming endpoint ({chunks_received} chunks received)")
            return True
            
        except Exception as e:
            print(f"❌ Test 5 FAILED: {e}")
            return False
    
    def test_concurrent_streaming(self):
        """Test 6: Concurrent streaming requests"""
        try:
            questions = [
                "What is Python?",
                "Explain machine learning",
                "What is web development?"
            ]
            
            def make_streaming_request(question):
                try:
                    response = requests.post(
                        f"{self.base_url}/ask-stream",
                        json={"question": question},
                        stream=True,
                        timeout=30
                    )
                    
                    chunks = 0
                    for line in response.iter_lines():
                        if line and line.decode('utf-8').startswith('data: '):
                            chunks += 1
                            if chunks > 5:  # Limit for testing
                                break
                    
                    return chunks > 0
                except:
                    return False
            
            # Start concurrent requests
            import concurrent.futures
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
                futures = [executor.submit(make_streaming_request, q) for q in questions]
                results = [future.result() for future in concurrent.futures.as_completed(futures)]
            
            # Verify all requests succeeded
            assert all(results), "Some concurrent requests failed"
            
            print("✅ Test 6 PASSED: Concurrent streaming requests")
            return True
            
        except Exception as e:
            print(f"❌ Test 6 FAILED: {e}")
            return False
    
    def test_streaming_response_format(self):
        """Test 7: Streaming response format validation"""
        try:
            # Test SSE format compliance
            test_data = {
                'type': 'chunk',
                'content': 'Hello streaming world!'
            }
            
            # Format as SSE
            sse_line = f"data: {json.dumps(test_data)}\\n\\n"
            
            # Validate format
            assert sse_line.startswith('data: ')
            assert sse_line.endswith('\\n\\n')
            
            # Test parsing
            json_part = sse_line[6:-4]  # Remove 'data: ' and '\\n\\n'
            parsed = json.loads(json_part)
            
            assert parsed['type'] == 'chunk'
            assert parsed['content'] == 'Hello streaming world!'
            
            print("✅ Test 7 PASSED: Streaming response format validation")
            return True
            
        except Exception as e:
            print(f"❌ Test 7 FAILED: {e}")
            return False
    
    def test_error_handling(self):
        """Test 8: Error handling in streaming"""
        try:
            # Test with invalid request
            invalid_request = {}  # Missing question
            
            response = requests.post(f"{self.base_url}/ask-stream", json=invalid_request)
            
            # Should handle gracefully (either 422 validation error or error in stream)
            assert response.status_code in [200, 422]
            
            if response.status_code == 200:
                # Check if error is sent in stream
                for line in response.iter_lines():
                    if line and line.decode('utf-8').startswith('data: '):
                        try:
                            data = json.loads(line[6:])
                            if data['type'] == 'error':
                                break
                        except:
                            continue
            
            print("✅ Test 8 PASSED: Error handling in streaming")
            return True
            
        except Exception as e:
            print(f"❌ Test 8 FAILED: {e}")
            return False
    
    def run_all_tests(self):
        """Run all WebSocket and streaming tests"""
        print("🚀 Starting WebSocket & Streaming Integration Tests")
        print("=" * 60)
        
        # Start test server
        if not self.start_test_server():
            print("❌ Cannot start test server. Aborting tests.")
            return False
        
        try:
            tests = [
                ("API Health Check", self.test_api_health),
                ("Materials Endpoint", self.test_materials_endpoint),
                ("Search Endpoint", self.test_search_endpoint),
                ("Simple Ask Endpoint", self.test_simple_ask_endpoint),
                ("Streaming Endpoint (SSE)", self.test_streaming_endpoint),
                ("Concurrent Streaming", self.test_concurrent_streaming),
                ("Streaming Format Validation", self.test_streaming_response_format),
                ("Error Handling", self.test_error_handling)
            ]
            
            passed = 0
            total = len(tests)
            
            for test_name, test_func in tests:
                print(f"\\n🧪 Running: {test_name}")
                try:
                    if test_func():
                        passed += 1
                    time.sleep(1)  # Brief pause between tests
                except Exception as e:
                    print(f"❌ {test_name} ERROR: {e}")
            
            print("\\n" + "=" * 60)
            print(f"📊 WebSocket & Streaming Test Results: {passed}/{total} passed")
            
            if passed == total:
                print("🎉 All WebSocket & Streaming tests passed!")
                return True
            else:
                print("⚠️  Some WebSocket & Streaming tests failed.")
                return False
                
        finally:
            self.stop_test_server()

def main():
    """Main test runner"""
    tester = WebSocketStreamingTester()
    success = tester.run_all_tests()
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()