"""
Movie Retrieval-Augmented Generation (RAG) System with Function Calling

This module implements an advanced RAG system that uses LLM function calling
to intelligently decide when to access the movie database. The system:

1. Uses LLM reasoning to determine when database access is needed
2. Maintains full conversation context for better responses
3. Provides conversation management with new conversation capabilities
4. Supports multiple LLM providers with function calling
"""

import os
import json
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
import Utils
from movie_vector_db import MovieVectorDB

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Function definition for LLM function calling
MOVIE_SEARCH_FUNCTION = {
    "name": "search_movie_database",
    "description": "Search the movie database for relevant movies based on user queries. Use this when the user asks about specific movies, genres, directors, actors, or wants movie recommendations.",
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The search query to find relevant movies"
            },
            "num_results": {
                "type": "integer",
                "description": "Number of movies to retrieve (default: 5)",
                "default": 5
            }
        },
        "required": ["query"]
    }
}


class MovieRAGSystem:
    """
    Advanced RAG system using LLM function calling for intelligent database access.

    This system uses the LLM's reasoning capabilities to decide when to access
    the movie database, providing more intelligent and context-aware responses.
    """

    def __init__(
        self,
        vector_db_path: Optional[str] = None,
        llm_provider: str = "mistral",
        llm_model: str = "mistral-small",
        embedding_model: str = "all-MiniLM-L6-v2",
        api_key: Optional[str] = None
    ):
        """
        Initialize the RAG system with function calling capabilities.

        Args:
            vector_db_path: Path to the vector database
            llm_provider: LLM provider (only 'mistral' supported)
            llm_model: Specific Mistral model name
            embedding_model: Sentence transformer model for embeddings
            api_key: Mistral API key
        """
        self.vector_db_path = vector_db_path
        self.llm_provider = llm_provider
        self.llm_model = llm_model
        self.api_key = api_key or self._get_api_key()

        # Initialize vector database
        self.vector_db = MovieVectorDB(model_name=embedding_model)
        if vector_db_path:
            self.load_vector_db(vector_db_path)

        # Conversation management
        self.conversation_history = []
        self.conversation_id = self._generate_conversation_id()

        # Function calling setup
        self.available_functions = {
            "search_movie_database": self._search_movie_database
        }

    def _get_api_key(self) -> str:
        """Get Mistral API key from environment variables."""
        print("00000000000000000000000000000000000000000000000000000000000000000000")
        if self.llm_provider != "mistral":
            raise ValueError("Only Mistral provider is supported")
        api_key = Utils.mistral_api_key #os.environ.get("MISTRAL_API_KEY")
        print('api_key: ',api_key )
        if not api_key:
            logger.error("MISTRAL_API_KEY environment variable not set")
            raise ValueError("MISTRAL_API_KEY environment variable is required")

        return api_key

    def _generate_conversation_id(self) -> str:
        """Generate a unique conversation ID."""
        return f"conv_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    def _search_movie_database(self, query: str, num_results: int = 5) -> List[Dict[str, Any]]:
        """
        Function called by LLM to search the movie database.

        Args:
            query: Search query for movies
            num_results: Number of results to return

        Returns:
            List of movie information dictionaries
        """
        if self.vector_db.index is None:
            logger.error("Vector database not loaded")
            return []

        try:
            results = self.vector_db.search(query, k=num_results)
            logger.info(f"Database search: '{query}' returned {len(results)} results")
            return results
        except Exception as e:
            logger.error(f"Error searching database: {e}")
            return []

    def load_vector_db(self, path: str) -> bool:
        """
        Load the vector database from the specified path.

        Args:
            path: Path to the vector database

        Returns:
            True if loaded successfully
        """
        try:
            success = self.vector_db.load(path)
            if success:
                logger.info(f"Vector database loaded successfully from {path}")
                return True
            else:
                logger.error(f"Failed to load vector database from {path}")
                return False
        except Exception as e:
            logger.error(f"Error loading vector database: {e}")
            return False

    def setup_vector_db(self, data_path: str, data_source: str = "wiki") -> bool:
        """
        Set up the vector database from movie data.

        Args:
            data_path: Path to the movie data CSV file
            data_source: Source of the data ('wiki' or 'tmdb')

        Returns:
            True if setup successfully
        """
        try:
            # Load and process data
            self.vector_db.load_data(data_path, data_source)
            self.vector_db.preprocess_data()
            self.vector_db.create_embeddings()
            self.vector_db.build_index()

            # Save the vector database
            save_path = self.vector_db.save()
            self.vector_db_path = save_path

            logger.info("Vector database setup completed successfully")
            return True

        except Exception as e:
            logger.error(f"Error setting up vector database: {e}")
            return False

    def _call_llm_with_functions(self, messages: List[Dict[str, str]]) -> Dict[str, Any]:
        """
        Call Mistral LLM with function calling capabilities.

        Args:
            messages: List of conversation messages

        Returns:
            LLM response with potential function calls
        """
        try:
            if self.llm_provider != "mistral":
                raise ValueError("Only Mistral provider is supported")

            return self._call_mistral_with_functions(messages)
        except Exception as e:
            logger.error(f"Error calling Mistral LLM: {e}")
            return {
                "content": f"I apologize, but I encountered an error: {str(e)}",
                "function_calls": []
            }

    def _call_mistral_with_functions(self, messages: List[Dict[str, str]]) -> Dict[str, Any]:
        """Call Mistral API with function calling."""
        try:
            from mistralai import Mistral

            # Validate API key
            if not self.api_key or self.api_key.strip() == "":
                raise ValueError("Mistral API key is empty or not set")

            client = Mistral(api_key=self.api_key)

            # Convert function definition to Mistral format
            tools = [{
                "type": "function",
                "function": MOVIE_SEARCH_FUNCTION
            }]

            response = client.chat.complete(
                model=self.llm_model,
                messages=messages,
                tools=tools,
                tool_choice="auto",
                temperature=0.7
            )

            message = response.choices[0].message

            result = {
                "content": message.content or "",
                "function_calls": []
            }

            if message.tool_calls:
                for tool_call in message.tool_calls:
                    if tool_call.function:
                        result["function_calls"].append({
                            "name": tool_call.function.name,
                            "arguments": json.loads(tool_call.function.arguments)
                        })

            return result

        except Exception as e:
            logger.error(f"Mistral API error: {e}")
            raise

    def _execute_function_calls(self, function_calls: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Execute function calls and return results.

        Args:
            function_calls: List of function calls to execute

        Returns:
            List of function call results
        """
        results = []

        for func_call in function_calls:
            func_name = func_call["name"]
            func_args = func_call["arguments"]

            if func_name in self.available_functions:
                try:
                    result = self.available_functions[func_name](**func_args)
                    results.append({
                        "function_name": func_name,
                        "arguments": func_args,
                        "result": result
                    })
                    logger.info(f"Executed function {func_name} with {len(result)} results")
                except Exception as e:
                    logger.error(f"Error executing function {func_name}: {e}")
                    results.append({
                        "function_name": func_name,
                        "arguments": func_args,
                        "result": [],
                        "error": str(e)
                    })
            else:
                logger.warning(f"Unknown function: {func_name}")

        return results

    def _format_conversation_messages(self, user_query: str) -> List[Dict[str, str]]:
        """
        Format the full conversation history for the LLM.

        Args:
            user_query: Current user query

        Returns:
            List of formatted messages
        """
        messages = [{
            "role": "system",
            "content": """You are a knowledgeable movie expert assistant. You help users find information about movies, provide recommendations, and answer questions.

When users ask about specific movies, genres, directors, actors, or want recommendations, use the search_movie_database function to find relevant information from the movie database.

For general conversation, greetings, or questions that don't require movie data, respond directly without using the function.

Be conversational, helpful, and provide detailed responses based on the information you find."""
        }]

        # Add conversation history
        for turn in self.conversation_history:
            messages.append({"role": "user", "content": turn["user_message"]})
            messages.append({"role": "assistant", "content": turn["assistant_message"]})

        # Add current user query
        messages.append({"role": "user", "content": user_query})

        return messages

    def ask(self, user_query: str) -> Dict[str, Any]:
        """
        Process user query using function calling to intelligently access the database.

        Args:
            user_query: User's question or request

        Returns:
            Dictionary containing response and metadata
        """
        start_time = datetime.now()

        try:
            # Format conversation messages
            messages = self._format_conversation_messages(user_query)

            # Call LLM with function calling
            llm_response = self._call_llm_with_functions(messages)

            # Execute any function calls
            function_results = []
            if llm_response["function_calls"]:
                function_results = self._execute_function_calls(llm_response["function_calls"])

                # If functions were called, make a second LLM call with results
                if function_results:
                    # Add function results to messages
                    for func_result in function_results:
                        function_message = {
                            "role": "function",
                            "name": func_result["function_name"],
                            "content": json.dumps(func_result["result"][:3])  # Limit to top 3 results
                        }
                        messages.append(function_message)

                    # Get final response with function results
                    final_response = self._call_llm_with_functions(messages)
                    response_text = final_response["content"]
                else:
                    response_text = llm_response["content"]
            else:
                response_text = llm_response["content"]

            # Prepare result
            result = {
                "query": user_query,
                "response": response_text,
                "function_calls": llm_response["function_calls"],
                "function_results": function_results,
                "database_accessed": len(function_results) > 0,
                "num_retrieved": sum(len(fr.get("result", [])) for fr in function_results),
                "conversation_id": self.conversation_id,
                "timestamp": start_time.isoformat(),
                "processing_time": (datetime.now() - start_time).total_seconds()
            }

            # Update conversation history
            self.conversation_history.append({
                "user_message": user_query,
                "assistant_message": response_text,
                "function_calls": llm_response["function_calls"],
                "timestamp": start_time.isoformat()
            })

            # Keep conversation history manageable (last 20 turns)
            if len(self.conversation_history) > 20:
                self.conversation_history = self.conversation_history[-20:]

            logger.info(f"Query processed in {result['processing_time']:.2f}s, DB accessed: {result['database_accessed']}")
            return result

        except Exception as e:
            logger.error(f"Error processing query '{user_query}': {e}")
            return {
                "query": user_query,
                "response": f"I apologize, but I encountered an error while processing your question: {str(e)}",
                "function_calls": [],
                "function_results": [],
                "database_accessed": False,
                "num_retrieved": 0,
                "conversation_id": self.conversation_id,
                "timestamp": start_time.isoformat(),
                "processing_time": (datetime.now() - start_time).total_seconds(),
                "error": str(e)
            }

    def start_new_conversation(self) -> str:
        """
        Start a new conversation by clearing history and generating new ID.

        Returns:
            New conversation ID
        """
        self.conversation_history = []
        self.conversation_id = self._generate_conversation_id()
        logger.info(f"Started new conversation: {self.conversation_id}")
        return self.conversation_id

    def get_conversation_history(self) -> List[Dict[str, Any]]:
        """Get the current conversation history."""
        return self.conversation_history.copy()

    def get_conversation_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the current conversation.

        Returns:
            Dictionary with conversation statistics
        """
        total_turns = len(self.conversation_history)
        db_accesses = sum(1 for turn in self.conversation_history if turn.get("function_calls"))

        return {
            "conversation_id": self.conversation_id,
            "total_turns": total_turns,
            "database_accesses": db_accesses,
            "efficiency": round((total_turns - db_accesses) / max(total_turns, 1) * 100, 1),
            "started_at": self.conversation_history[0]["timestamp"] if self.conversation_history else None
        }


# Example usage
if __name__ == "__main__":
    print("Initializing Movie RAG System with Function Calling...")

    # Initialize the RAG system
    rag_system = MovieRAGSystem(
        llm_provider="mistral",
        llm_model="mistral-small"
    )

    # Load vector database (if available)
    vector_db_path = "saved_models/vector_db"
    if os.path.exists(os.path.join(vector_db_path, "faiss_index.bin")):
        rag_system.load_vector_db(vector_db_path)
        print("✅ Vector database loaded")
    else:
        print("❌ Vector database not found. Please set it up first.")
        exit(1)

    # Example questions
    example_questions = [
        "Hello! How are you?",  # Should not access database
        "What are some good sci-fi movies?",  # Should access database
        "Tell me more about the first one",  # Should use conversation context
        "Who directed it?",  # Should use conversation context
        "What are some comedy movies?",  # Should access database again
    ]

    print("\n🎬 Testing Function Calling RAG System:")
    print("=" * 50)

    for i, question in enumerate(example_questions, 1):
        print(f"\n{i}. Question: {question}")
        result = rag_system.ask(question)

        print(f"   Response: {result['response'][:100]}...")
        print(f"   Database accessed: {result['database_accessed']}")
        print(f"   Function calls: {len(result['function_calls'])}")
        print(f"   Processing time: {result['processing_time']:.2f}s")

    # Show conversation stats
    print(f"\n📊 Conversation Statistics:")
    stats = rag_system.get_conversation_stats()
    for key, value in stats.items():
        print(f"   {key}: {value}")

    print("\n✨ Function calling RAG system ready!")
    print("The LLM intelligently decides when to access the database.")
    print("Use start_new_conversation() to begin a fresh conversation.")
