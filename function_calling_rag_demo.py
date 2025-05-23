"""
Function Calling RAG System Demo

This script demonstrates the enhanced Movie RAG System that uses LLM function calling
to intelligently decide when to access the database. Key features:

1. LLM reasoning determines when database access is needed
2. Full conversation context is maintained
3. Intelligent function calling reduces unnecessary database queries
4. New conversation management with conversation IDs
"""

import os
import time
from movie_rag_system import MovieRAGSystem

def print_section_header(title):
    """Print a formatted section header."""
    print("\n" + "=" * 60)
    print(f"🎯 {title}")
    print("=" * 60)

def check_setup():
    """Check if the system is properly set up."""
    print("🔍 Checking system setup...")

    # Check for vector database
    vector_db_path = "saved_models/vector_db"
    if not os.path.exists(os.path.join(vector_db_path, "faiss_index.bin")):
        print("❌ Vector database not found")
        print(f"   Expected location: {vector_db_path}")
        print("   Please run setup_rag_system.py first")
        return False

    # Check for Mistral API key
    mistral_key = os.environ.get('MISTRAL_API_KEY')

    if not mistral_key:
        print("⚠️  MISTRAL_API_KEY environment variable not set")
        print("   Please set your Mistral API key:")
        print("   export MISTRAL_API_KEY='your-api-key-here'")
        print("   The demo will continue but may not work properly")
    else:
        print("✅ Mistral API key found")

    print("✅ Vector database found")
    return True

def initialize_rag_system():
    """Initialize the RAG system."""
    print("\n🚀 Initializing Function Calling RAG System...")

    # Use Mistral provider
    if not os.environ.get('MISTRAL_API_KEY'):
        print("❌ MISTRAL_API_KEY not found. Please set your Mistral API key.")
        return None

    provider, model = "mistral", "mistral-small"

    print(f"🤖 Using {provider.upper()} with model {model}")

    try:
        rag_system = MovieRAGSystem(
            llm_provider=provider,
            llm_model=model
        )

        # Load vector database
        success = rag_system.load_vector_db("saved_models/vector_db")
        if success:
            print("✅ RAG system initialized successfully!")
            return rag_system
        else:
            print("❌ Failed to load vector database")
            return None

    except Exception as e:
        print(f"❌ Error initializing RAG system: {e}")
        return None

def demonstrate_intelligent_function_calling(rag_system):
    """Demonstrate how the LLM intelligently decides when to call functions."""
    print_section_header("Intelligent Function Calling")

    test_queries = [
        {
            "query": "Hello! How are you today?",
            "expected": "No function call - greeting",
            "should_call_db": False
        },
        {
            "query": "What are some good science fiction movies?",
            "expected": "Function call - movie search needed",
            "should_call_db": True
        },
        {
            "query": "Thank you for the recommendations!",
            "expected": "No function call - acknowledgment",
            "should_call_db": False
        },
        {
            "query": "Tell me more about the first movie you mentioned",
            "expected": "No function call - uses conversation context",
            "should_call_db": False
        },
        {
            "query": "What are some good horror movies from the 1980s?",
            "expected": "Function call - new search needed",
            "should_call_db": True
        }
    ]

    print("Testing LLM's decision-making for database access:")
    print("-" * 50)

    for i, test in enumerate(test_queries, 1):
        print(f"\n{i}. Query: '{test['query']}'")
        print(f"   Expected: {test['expected']}")

        start_time = time.time()
        result = rag_system.ask(test['query'])
        end_time = time.time()

        # Check if the LLM made the right decision
        db_accessed = result['database_accessed']
        correct_decision = db_accessed == test['should_call_db']

        status = "✅ CORRECT" if correct_decision else "❌ INCORRECT"
        db_icon = "🔍" if db_accessed else "💭"

        print(f"   Result: {status} - {db_icon} DB accessed: {db_accessed}")
        print(f"   Function calls: {len(result['function_calls'])}")
        print(f"   Response: {result['response'][:80]}...")
        print(f"   Time: {end_time - start_time:.2f}s")

def demonstrate_conversation_context(rag_system):
    """Demonstrate how conversation context is maintained."""
    print_section_header("Conversation Context Management")

    # Start a new conversation
    conv_id = rag_system.start_new_conversation()
    print(f"🆕 Started new conversation: {conv_id}")

    conversation_flow = [
        "What are some popular action movies?",
        "Which of those movies has the best special effects?",
        "Who directed that movie?",
        "What other movies did that director make?",
        "When was the first movie released?"
    ]

    print("\n🗣️  Conversation Flow:")
    print("-" * 30)

    for i, query in enumerate(conversation_flow, 1):
        print(f"\nTurn {i}: {query}")

        result = rag_system.ask(query)

        db_icon = "🔍" if result['database_accessed'] else "💭"
        print(f"   {db_icon} Database accessed: {result['database_accessed']}")
        print(f"   Response: {result['response'][:100]}...")

        # Show conversation stats
        stats = rag_system.get_conversation_stats()
        print(f"   📊 Total turns: {stats['total_turns']}, DB accesses: {stats['database_accesses']}")

def demonstrate_new_conversation_feature(rag_system):
    """Demonstrate the new conversation management feature."""
    print_section_header("New Conversation Management")

    # Show current conversation stats
    stats = rag_system.get_conversation_stats()
    print(f"📊 Current conversation stats:")
    print(f"   ID: {stats['conversation_id']}")
    print(f"   Turns: {stats['total_turns']}")
    print(f"   DB accesses: {stats['database_accesses']}")
    print(f"   Efficiency: {stats['efficiency']}%")

    # Ask a question in current conversation
    print(f"\n🗣️  Asking in current conversation...")
    result1 = rag_system.ask("What are some comedy movies?")
    print(f"   Response: {result1['response'][:80]}...")
    print(f"   DB accessed: {result1['database_accessed']}")

    # Start new conversation
    print(f"\n🆕 Starting new conversation...")
    new_conv_id = rag_system.start_new_conversation()
    print(f"   New conversation ID: {new_conv_id}")

    # Ask a follow-up question that would have used context in old conversation
    print(f"\n🗣️  Asking follow-up in new conversation...")
    result2 = rag_system.ask("Tell me more about the first one")
    print(f"   Response: {result2['response'][:80]}...")
    print(f"   DB accessed: {result2['database_accessed']}")

    # Show new conversation stats
    new_stats = rag_system.get_conversation_stats()
    print(f"\n📊 New conversation stats:")
    print(f"   ID: {new_stats['conversation_id']}")
    print(f"   Turns: {new_stats['total_turns']}")
    print(f"   DB accesses: {new_stats['database_accesses']}")

def demonstrate_efficiency_comparison(rag_system):
    """Compare efficiency with and without conversation context."""
    print_section_header("Efficiency Comparison")

    # Start fresh conversation
    rag_system.start_new_conversation()

    # Scenario 1: Questions that should reuse context
    print("📈 Scenario 1: Context Reuse")
    context_questions = [
        "What are some thriller movies?",
        "Which of those is the most suspenseful?",
        "Who directed it?",
        "What's the plot about?"
    ]

    for i, question in enumerate(context_questions, 1):
        result = rag_system.ask(question)
        db_icon = "🔍" if result['database_accessed'] else "💭"
        print(f"   {i}. {db_icon} '{question}' -> DB: {result['database_accessed']}")

    stats1 = rag_system.get_conversation_stats()
    print(f"   📊 Efficiency: {stats1['efficiency']}% ({stats1['database_accesses']}/{stats1['total_turns']} DB calls)")

    # Scenario 2: Independent questions (new conversation each time)
    print(f"\n📈 Scenario 2: Independent Questions")
    independent_questions = [
        "What are some drama movies?",
        "What are some comedy movies?",
        "What are some horror movies?",
        "What are some romance movies?"
    ]

    for i, question in enumerate(independent_questions, 1):
        rag_system.start_new_conversation()  # Fresh context each time
        result = rag_system.ask(question)
        db_icon = "🔍" if result['database_accessed'] else "💭"
        print(f"   {i}. {db_icon} '{question}' -> DB: {result['database_accessed']}")

def interactive_demo(rag_system):
    """Interactive demo mode."""
    print_section_header("Interactive Demo Mode")

    print("🎮 Interactive Mode - Try the function calling system yourself!")
    print("Commands:")
    print("  'new' - Start new conversation")
    print("  'stats' - Show conversation statistics")
    print("  'quit' - Exit interactive mode")
    print()

    while True:
        try:
            user_input = input("🎬 Your question: ").strip()

            if user_input.lower() == 'quit':
                print("👋 Exiting interactive mode")
                break
            elif user_input.lower() == 'new':
                conv_id = rag_system.start_new_conversation()
                print(f"🆕 Started new conversation: {conv_id[:12]}...")
                continue
            elif user_input.lower() == 'stats':
                stats = rag_system.get_conversation_stats()
                print(f"📊 Conversation Stats:")
                for key, value in stats.items():
                    print(f"   {key}: {value}")
                continue
            elif not user_input:
                continue

            print("🤔 Processing...")
            start_time = time.time()
            result = rag_system.ask(user_input)
            end_time = time.time()

            db_icon = "🔍" if result['database_accessed'] else "💭"
            print(f"\n🤖 {result['response']}")
            print(f"\n📊 {db_icon} DB: {result['database_accessed']} | ⏱️ {end_time - start_time:.2f}s | 🔧 Functions: {len(result['function_calls'])}")
            print()

        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

def main():
    """Main demo function."""
    print("🎬 Function Calling RAG System Demo")
    print("This demo shows how LLMs intelligently decide when to access the database")

    # Check setup
    if not check_setup():
        return

    # Initialize system
    rag_system = initialize_rag_system()
    if not rag_system:
        return

    try:
        # Run demonstrations
        demonstrate_intelligent_function_calling(rag_system)
        demonstrate_conversation_context(rag_system)
        demonstrate_new_conversation_feature(rag_system)
        demonstrate_efficiency_comparison(rag_system)

        # Ask if user wants interactive mode
        print("\n" + "=" * 60)
        response = input("Would you like to try interactive mode? (y/n): ").strip().lower()
        if response in ['y', 'yes']:
            interactive_demo(rag_system)

        # Final summary
        print("\n🎉 Demo completed!")
        print("Key benefits of function calling RAG:")
        print("  • LLM intelligently decides when to access database")
        print("  • Full conversation context maintained")
        print("  • Reduced unnecessary database queries")
        print("  • Better conversational experience")
        print("  • Easy conversation management")

    except KeyboardInterrupt:
        print("\n👋 Demo interrupted. Goodbye!")
    except Exception as e:
        print(f"\n❌ Demo error: {e}")

if __name__ == "__main__":
    main()
