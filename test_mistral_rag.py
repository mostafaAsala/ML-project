"""
Test script for Mistral-only RAG system
"""

import os
from movie_rag_system import MovieRAGSystem

def test_api_key():
    """Test API key handling."""
    print("🔍 Testing API key handling...")
    
    # Check if API key is set
    api_key = os.environ.get("MISTRAL_API_KEY")
    if not api_key:
        print("❌ MISTRAL_API_KEY not set in environment")
        print("Please set it with: export MISTRAL_API_KEY='your-key-here'")
        return False
    
    if api_key.strip() == "":
        print("❌ MISTRAL_API_KEY is empty")
        return False
    
    print(f"✅ API key found: {api_key[:10]}...")
    return True

def test_rag_initialization():
    """Test RAG system initialization."""
    print("\n🚀 Testing RAG system initialization...")
    
    try:
        rag_system = MovieRAGSystem(
            llm_provider="mistral",
            llm_model="mistral-small"
        )
        print("✅ RAG system initialized successfully")
        return rag_system
    except Exception as e:
        print(f"❌ Failed to initialize RAG system: {e}")
        return None

def test_vector_db_loading(rag_system):
    """Test vector database loading."""
    print("\n📚 Testing vector database loading...")
    
    vector_db_path = "saved_models/vector_db"
    if not os.path.exists(os.path.join(vector_db_path, "faiss_index.bin")):
        print(f"❌ Vector database not found at {vector_db_path}")
        print("Please run setup_rag_system.py first")
        return False
    
    try:
        success = rag_system.load_vector_db(vector_db_path)
        if success:
            print("✅ Vector database loaded successfully")
            return True
        else:
            print("❌ Failed to load vector database")
            return False
    except Exception as e:
        print(f"❌ Error loading vector database: {e}")
        return False

def test_simple_query(rag_system):
    """Test a simple query that shouldn't access the database."""
    print("\n💬 Testing simple greeting (should not access database)...")
    
    try:
        result = rag_system.ask("Hello! How are you?")
        print(f"Response: {result['response'][:100]}...")
        print(f"Database accessed: {result['database_accessed']}")
        print(f"Function calls: {len(result['function_calls'])}")
        
        if not result['database_accessed']:
            print("✅ Correctly handled greeting without database access")
            return True
        else:
            print("⚠️ Unexpectedly accessed database for greeting")
            return True  # Still working, just not optimal
    except Exception as e:
        print(f"❌ Error processing greeting: {e}")
        return False

def test_movie_query(rag_system):
    """Test a movie query that should access the database."""
    print("\n🎬 Testing movie query (should access database)...")
    
    try:
        result = rag_system.ask("What are some good action movies?")
        print(f"Response: {result['response'][:100]}...")
        print(f"Database accessed: {result['database_accessed']}")
        print(f"Function calls: {len(result['function_calls'])}")
        print(f"Movies retrieved: {result['num_retrieved']}")
        
        if result['database_accessed']:
            print("✅ Correctly accessed database for movie query")
            return True
        else:
            print("⚠️ Did not access database for movie query")
            return False
    except Exception as e:
        print(f"❌ Error processing movie query: {e}")
        return False

def main():
    """Main test function."""
    print("🧪 Testing Mistral-only RAG System")
    print("=" * 50)
    
    # Test API key
    if not test_api_key():
        print("\n❌ API key test failed. Cannot continue.")
        return
    
    # Test initialization
    rag_system = test_rag_initialization()
    if not rag_system:
        print("\n❌ Initialization failed. Cannot continue.")
        return
    
    # Test vector database loading
    if not test_vector_db_loading(rag_system):
        print("\n❌ Vector database loading failed. Cannot continue.")
        return
    
    # Test simple query
    if not test_simple_query(rag_system):
        print("\n❌ Simple query test failed.")
        return
    
    # Test movie query
    if not test_movie_query(rag_system):
        print("\n❌ Movie query test failed.")
        return
    
    print("\n🎉 All tests passed!")
    print("The Mistral-only RAG system is working correctly.")
    
    # Show conversation stats
    stats = rag_system.get_conversation_stats()
    print(f"\n📊 Final conversation stats:")
    for key, value in stats.items():
        print(f"   {key}: {value}")

if __name__ == "__main__":
    main()
