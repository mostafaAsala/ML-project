"""
Movie RAG System Demo

A simple demonstration script that showcases the Movie RAG System capabilities.
This script provides an interactive demo that can be run without requiring
a full vector database setup.
"""

import os
import sys
from datetime import datetime
from movie_rag_system import MovieRAGSystem

def print_header():
    """Print a nice header for the demo."""
    print("=" * 60)
    print("🎬 MOVIE RAG SYSTEM DEMO")
    print("=" * 60)
    print("Welcome to the Movie RAG System demonstration!")
    print("This system combines vector search with AI to answer movie questions.")
    print()

def check_prerequisites():
    """Check if the system is ready to run."""
    print("🔍 Checking prerequisites...")
    
    # Check for vector database
    vector_db_path = "saved_models/vector_db"
    if os.path.exists(os.path.join(vector_db_path, "faiss_index.bin")):
        print("✅ Vector database found")
        return True, vector_db_path
    else:
        print("❌ Vector database not found")
        print(f"   Expected location: {vector_db_path}")
        print("   Please run the vector database setup first.")
        return False, None

def setup_rag_system(vector_db_path):
    """Set up the RAG system with appropriate configuration."""
    print("\n🚀 Initializing RAG system...")
    
    # Check for API keys
    api_providers = {
        'MISTRAL_API_KEY': 'mistral',
        'OPENAI_API_KEY': 'openai',
        'ANTHROPIC_API_KEY': 'anthropic'
    }
    
    selected_provider = None
    for key, provider in api_providers.items():
        if os.environ.get(key):
            selected_provider = provider
            print(f"✅ Found {provider.upper()} API key")
            break
    
    if not selected_provider:
        print("⚠️  No API keys found. Using mock responses for demo.")
        selected_provider = "mistral"  # Default for demo
    
    # Initialize RAG system
    try:
        rag_system = MovieRAGSystem(
            llm_provider=selected_provider,
            llm_model="mistral-small" if selected_provider == "mistral" else "gpt-3.5-turbo",
            top_k_retrieval=5
        )
        
        # Load vector database
        success = rag_system.load_vector_db(vector_db_path)
        if success:
            print("✅ RAG system initialized successfully!")
            return rag_system
        else:
            print("❌ Failed to load vector database")
            return None
            
    except Exception as e:
        print(f"❌ Error initializing RAG system: {e}")
        return None

def run_demo_questions(rag_system):
    """Run a series of demo questions."""
    print("\n🎯 Running demo questions...")
    print("=" * 40)
    
    demo_questions = [
        {
            "question": "What are some good science fiction movies?",
            "description": "General genre recommendation"
        },
        {
            "question": "Tell me about movies with time travel themes",
            "description": "Specific theme search"
        },
        {
            "question": "What movies are similar to The Matrix?",
            "description": "Similarity-based recommendation"
        },
        {
            "question": "Can you recommend some classic films from the 1970s?",
            "description": "Era-based recommendation"
        },
        {
            "question": "What are some underrated horror movies?",
            "description": "Genre with quality filter"
        }
    ]
    
    for i, demo in enumerate(demo_questions, 1):
        print(f"\n📝 Demo Question {i}: {demo['description']}")
        print(f"❓ {demo['question']}")
        print("-" * 50)
        
        try:
            # Ask the question
            start_time = datetime.now()
            result = rag_system.ask(demo['question'])
            end_time = datetime.now()
            
            # Display results
            print(f"🤖 Answer: {result['response']}")
            print(f"📊 Retrieved {result['num_retrieved']} relevant movies")
            print(f"⏱️  Response time: {result['processing_time']:.2f} seconds")
            
            # Show top retrieved movies if available
            if result.get('retrieved_docs'):
                print("\n🎬 Top retrieved movies:")
                for j, doc in enumerate(result['retrieved_docs'][:3], 1):
                    title = doc.get('title', 'Unknown')
                    year = doc.get('year', 'Unknown')
                    score = doc.get('similarity_score', 0)
                    print(f"   {j}. {title} ({year}) - Relevance: {score:.3f}")
            
        except Exception as e:
            print(f"❌ Error processing question: {e}")
        
        print()

def run_conversation_demo(rag_system):
    """Demonstrate conversational capabilities."""
    print("\n💬 Conversation Demo")
    print("=" * 40)
    print("This demonstrates how the system maintains context across questions.")
    
    conversation_flow = [
        "What are some good action movies?",
        "Which of those movies has the best special effects?",
        "Tell me more about the plot of that movie",
        "Who directed it?"
    ]
    
    for i, question in enumerate(conversation_flow, 1):
        print(f"\n🗣️  Turn {i}: {question}")
        print("-" * 30)
        
        try:
            result = rag_system.ask(question)
            print(f"🤖 {result['response']}")
            
            # Show conversation history length
            history = rag_system.get_conversation_history()
            print(f"📚 Conversation history: {len(history)} turns")
            
        except Exception as e:
            print(f"❌ Error: {e}")

def show_system_stats(rag_system):
    """Show system statistics and information."""
    print("\n📊 System Statistics")
    print("=" * 40)
    
    try:
        # Vector database stats
        if rag_system.vector_db.movies_df is not None:
            num_movies = len(rag_system.vector_db.movies_df)
            print(f"🎬 Movies in database: {num_movies:,}")
            
            # Show sample movie titles
            sample_titles = rag_system.vector_db.movies_df['title'].head(5).tolist()
            print(f"📝 Sample titles: {', '.join(sample_titles)}")
        
        # Configuration
        print(f"🤖 LLM Provider: {rag_system.llm_provider}")
        print(f"🔧 Model: {rag_system.llm_model}")
        print(f"🔍 Retrieval count: {rag_system.top_k_retrieval}")
        print(f"📏 Max context length: {rag_system.max_context_length}")
        
        # Conversation stats
        history = rag_system.get_conversation_history()
        print(f"💬 Conversation turns: {len(history)}")
        
    except Exception as e:
        print(f"❌ Error getting stats: {e}")

def interactive_mode(rag_system):
    """Run interactive mode for user questions."""
    print("\n🎮 Interactive Mode")
    print("=" * 40)
    print("Now you can ask your own questions!")
    print("Type 'quit' to exit, 'stats' for system info, 'clear' to clear history")
    print()
    
    while True:
        try:
            user_input = input("🎬 Your movie question: ").strip()
            
            if user_input.lower() == 'quit':
                print("👋 Thanks for using the Movie RAG System!")
                break
            elif user_input.lower() == 'stats':
                show_system_stats(rag_system)
                continue
            elif user_input.lower() == 'clear':
                rag_system.clear_conversation_history()
                print("🧹 Conversation history cleared!")
                continue
            elif not user_input:
                continue
            
            # Process the question
            print("🤔 Thinking...")
            result = rag_system.ask(user_input)
            
            print(f"\n🤖 Answer: {result['response']}")
            print(f"⏱️  ({result['processing_time']:.2f}s, {result['num_retrieved']} sources)")
            print()
            
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")
            print()

def main():
    """Main demo function."""
    print_header()
    
    # Check prerequisites
    ready, vector_db_path = check_prerequisites()
    if not ready:
        print("\n💡 To set up the vector database:")
        print("   1. Ensure you have movie data (CSV file)")
        print("   2. Run: python movie_vector_db.py")
        print("   3. Or use the setup in rag_example.py")
        return
    
    # Set up RAG system
    rag_system = setup_rag_system(vector_db_path)
    if not rag_system:
        print("\n❌ Failed to initialize RAG system. Please check your setup.")
        return
    
    # Show system stats
    show_system_stats(rag_system)
    
    # Run demos
    try:
        run_demo_questions(rag_system)
        run_conversation_demo(rag_system)
        
        # Ask user if they want interactive mode
        print("\n" + "=" * 60)
        response = input("Would you like to try interactive mode? (y/n): ").strip().lower()
        if response in ['y', 'yes']:
            interactive_mode(rag_system)
        else:
            print("👋 Demo completed! Thanks for trying the Movie RAG System.")
            
    except KeyboardInterrupt:
        print("\n👋 Demo interrupted. Goodbye!")
    except Exception as e:
        print(f"\n❌ Demo error: {e}")

if __name__ == "__main__":
    main()
