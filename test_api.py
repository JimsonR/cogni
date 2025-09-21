"""
Test script for the Vector Search API
Run this after starting the API server to test the search functionality
"""

import requests
import json
from typing import Dict, Any

# API base URL
API_BASE_URL = "http://localhost:8000"

def test_health():
    """Test the health endpoint."""
    print("🔍 Testing health endpoint...")
    try:
        response = requests.get(f"{API_BASE_URL}/health")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Health check passed: {data['status']}")
            print(f"   MongoDB: {'✅' if data['mongodb_connected'] else '❌'}")
            print(f"   OpenAI: {'✅' if data['openai_connected'] else '❌'}")
            print(f"   Documents: {data.get('collection_count', 'N/A')}")
            return True
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Health check error: {e}")
        return False

def test_search_post(query: str, limit: int = 3, threshold: float = 0.7):
    """Test the POST search endpoint."""
    print(f"\n🔍 Testing POST search with query: '{query}'")
    try:
        payload = {
            "query": query,
            "limit": limit,
            "threshold": threshold,
            "include_embeddings": False
        }
        
        response = requests.post(f"{API_BASE_URL}/search", json=payload)
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Search successful!")
            print(f"   Results: {data['total_results']}")
            print(f"   Execution time: {data['execution_time_ms']}ms")
            
            for i, result in enumerate(data['results'], 1):
                print(f"\n   Result {i} (Score: {result['similarity_score']:.3f}):")
                print(f"   Question: {result['question'][:100]}...")
                print(f"   Answer: {result['answer'][:150]}...")
                if result.get('follow_up_questions_and_answers'):
                    print(f"   Follow-ups: {len(result['follow_up_questions_and_answers'])} questions")
            
            return data
        else:
            print(f"❌ Search failed: {response.status_code}")
            print(f"   Error: {response.text}")
            return None
            
    except Exception as e:
        print(f"❌ Search error: {e}")
        return None

def test_search_get(query: str, limit: int = 3):
    """Test the GET search endpoint."""
    print(f"\n🔍 Testing GET search with query: '{query}'")
    try:
        params = {
            "q": query,
            "limit": limit,
            "threshold": 0.7
        }
        
        response = requests.get(f"{API_BASE_URL}/search", params=params)
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ GET Search successful!")
            print(f"   Results: {data['total_results']}")
            print(f"   Execution time: {data['execution_time_ms']}ms")
            return data
        else:
            print(f"❌ GET Search failed: {response.status_code}")
            return None
            
    except Exception as e:
        print(f"❌ GET Search error: {e}")
        return None

def main():
    """Run all tests."""
    print("🚀 Starting Vector Search API Tests")
    print("=" * 50)
    
    # Test health
    if not test_health():
        print("\n❌ Health check failed. Make sure the API server is running.")
        return
    
    # Test search with various queries
    test_queries = [
        "What was the contractor's request for extension?",
        "bidding requirements",
        "performance security",
        "payment terms",
        "project completion"
    ]
    
    for query in test_queries:
        test_search_post(query, limit=2, threshold=0.6)
    
    # Test GET endpoint
    test_search_get("project delays", limit=2)
    
    print("\n" + "=" * 50)
    print("🎉 Tests completed!")
    print("\nTo explore the API interactively, visit: http://localhost:8000/docs")

if __name__ == "__main__":
    main()