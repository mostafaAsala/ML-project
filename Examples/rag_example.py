"""
Movie RAG System Example

This script demonstrates how to use the Movie RAG System for question answering.
It includes examples of:
1. Setting up the vector database
2. Asking various types of questions
3. Handling conversation history
4. Different query types and use cases
"""

import os
import json
from movie_rag_system import MovieRAGSystem

def setup_rag_system_example():
    """
    Example of setting up the RAG system with a movie database.
    """
    print("=== Setting up Movie RAG System ===")
    
    # Initialize the RAG system
    rag_system = MovieRAGSystem(
        llm_provider="mistral",  # Change to your preferred provider
        llm_model="mistral-small",
        embedding_model="all-MiniLM-L6-v2",
        top_k_retrieval=5
    )
    
    # Check if vector database already exists
    vector_db_path = "saved_models/vector_db"
    if os.path.exists(os.path.join(vector_db_path, "faiss_index.bin")):
        print("Loading existing vector database...")
        success = rag_system.load_vector_db(vector_db_path)
        if success:
            print("✓ Vector database loaded successfully!")
        else:
            print("✗ Failed to load vector database")
            return None
    else:
        print("Setting up new vector database...")
        # You need to provide the path to your movie data
        data_path = "wiki_movie_plots_deduped_cleaned.csv"  # Update this path
        
        if os.path.exists(data_path):
            success = rag_system.setup_vector_db(data_path, data_source="wiki")
            if success:
                print("✓ Vector database setup completed!")
            else:
                print("✗ Failed to setup vector database")
                return None
        else:
            print(f"✗ Movie data file not found: {data_path}")
            print("Please provide the correct path to your movie data CSV file.")
            return None
    
    return rag_system

def basic_qa_example(rag_system):
    """
    Example of basic question-answering functionality.
    """
    print("\n=== Basic Question-Answering Examples ===")
    
    # Example questions
    questions = [
        "What are some good science fiction movies?",
        "Tell me about movies with time travel themes",
        "What movies are similar to Inception?",
        "Can you recommend some horror movies from the 1980s?",
        "What are the best romantic comedies?"
    ]
    
    for question in questions:
        print(f"\nQ: {question}")
        print("-" * 50)
        
        # Ask the question
        result = rag_system.ask(question)
        
        # Print the response
        print(f"A: {result['response']}")
        print(f"Retrieved {result['num_retrieved']} relevant movies")
        print(f"Processing time: {result['processing_time']:.2f} seconds")
        
        # Optionally show retrieved movies
        if result['retrieved_docs']:
            print("\nTop retrieved movies:")
            for i, doc in enumerate(result['retrieved_docs'][:3], 1):
                title = doc.get('title', 'Unknown')
                year = doc.get('year', 'Unknown')
                score = doc.get('similarity_score', 0)
                print(f"  {i}. {title} ({year}) - Score: {score:.3f}")

def conversation_example(rag_system):
    """
    Example of conversational interaction with context.
    """
    print("\n=== Conversational Example ===")
    
    # Simulate a conversation
    conversation = [
        "What are some good action movies?",
        "Which of those movies has the best special effects?",
        "Tell me more about the plot of that movie",
        "Who directed it?",
        "What other movies did that director make?"
    ]
    
    for i, question in enumerate(conversation, 1):
        print(f"\nTurn {i}: {question}")
        print("-" * 40)
        
        result = rag_system.ask(question)
        print(f"Response: {result['response']}")
        
        # Show conversation history length
        history = rag_system.get_conversation_history()
        print(f"Conversation history: {len(history)} turns")

def specific_movie_queries_example(rag_system):
    """
    Example of specific movie-related queries.
    """
    print("\n=== Specific Movie Queries ===")
    
    specific_queries = [
        "What is the plot of The Matrix?",
        "Who are the main actors in Pulp Fiction?",
        "What genre is Blade Runner?",
        "When was Casablanca released?",
        "What movies did Steven Spielberg direct in the 1990s?"
    ]
    
    for query in specific_queries:
        print(f"\nQuery: {query}")
        print("-" * 50)
        
        result = rag_system.ask(query, k=3)  # Retrieve fewer documents for specific queries
        print(f"Answer: {result['response']}")

def recommendation_example(rag_system):
    """
    Example of movie recommendation functionality.
    """
    print("\n=== Movie Recommendation Examples ===")
    
    recommendation_queries = [
        "I like movies with complex plots and mind-bending twists. What do you recommend?",
        "Recommend some feel-good family movies",
        "I enjoyed The Godfather. What similar movies should I watch?",
        "What are some underrated gems from the 1970s?",
        "I want to watch something funny but not too silly. Any suggestions?"
    ]
    
    for query in recommendation_queries:
        print(f"\nRequest: {query}")
        print("-" * 60)
        
        result = rag_system.ask(query, k=7)  # Get more results for recommendations
        print(f"Recommendations: {result['response']}")
        
        # Show the movies that influenced the recommendation
        if result['retrieved_docs']:
            print(f"\nBased on analysis of {len(result['retrieved_docs'])} movies in the database")

def save_and_load_conversation_example(rag_system):
    """
    Example of saving and loading conversation history.
    """
    print("\n=== Save/Load Conversation Example ===")
    
    # Have a short conversation
    questions = [
        "What are some classic movies everyone should watch?",
        "Which of those is your top recommendation?"
    ]
    
    for question in questions:
        result = rag_system.ask(question)
        print(f"Q: {question}")
        print(f"A: {result['response'][:100]}...")
    
    # Save conversation
    conversation_file = "sample_conversation.json"
    rag_system.save_conversation(conversation_file)
    print(f"\n✓ Conversation saved to {conversation_file}")
    
    # Clear and reload
    rag_system.clear_conversation_history()
    print("✓ Conversation history cleared")
    
    rag_system.load_conversation(conversation_file)
    print("✓ Conversation history reloaded")
    
    # Show loaded history
    history = rag_system.get_conversation_history()
    print(f"Loaded {len(history)} conversation turns")

def error_handling_example(rag_system):
    """
    Example of how the system handles various edge cases.
    """
    print("\n=== Error Handling Examples ===")
    
    edge_cases = [
        "",  # Empty query
        "What is the meaning of life?",  # Non-movie related
        "Tell me about a movie that doesn't exist: XYZ123",  # Non-existent movie
        "A" * 1000,  # Very long query
    ]
    
    for i, query in enumerate(edge_cases, 1):
        print(f"\nEdge case {i}: {query[:50]}{'...' if len(query) > 50 else ''}")
        result = rag_system.ask(query)
        print(f"Response: {result['response'][:100]}...")
        if 'error' in result:
            print(f"Error handled: {result['error']}")

def main():
    """
    Main function to run all examples.
    """
    print("Movie RAG System Examples")
    print("=" * 50)
    
    # Setup the RAG system
    rag_system = setup_rag_system_example()
    
    if rag_system is None:
        print("Failed to setup RAG system. Please check your configuration.")
        return
    
    try:
        # Run examples
        basic_qa_example(rag_system)
        conversation_example(rag_system)
        specific_movie_queries_example(rag_system)
        recommendation_example(rag_system)
        save_and_load_conversation_example(rag_system)
        error_handling_example(rag_system)
        
        print("\n" + "=" * 50)
        print("All examples completed successfully!")
        print("You can now use the RAG system for your own movie questions.")
        
    except Exception as e:
        print(f"Error running examples: {e}")
        print("Please check your API keys and configuration.")

def interactive_mode():
    """
    Interactive mode for testing the RAG system.
    """
    print("\n=== Interactive Mode ===")
    print("Type 'quit' to exit, 'clear' to clear conversation history")
    
    rag_system = setup_rag_system_example()
    if rag_system is None:
        return
    
    while True:
        try:
            query = input("\nYour question: ").strip()
            
            if query.lower() == 'quit':
                break
            elif query.lower() == 'clear':
                rag_system.clear_conversation_history()
                print("Conversation history cleared.")
                continue
            elif not query:
                continue
            
            result = rag_system.ask(query)
            print(f"\nAnswer: {result['response']}")
            print(f"(Retrieved {result['num_retrieved']} movies, took {result['processing_time']:.2f}s)")
            
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    # Run all examples
    main()
    
    # Uncomment the line below to run interactive mode
    # interactive_mode()
