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
from movie_vector_db import MovieVectorDB

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import dotenv
dotenv.load_dotenv()

"""
ALWAYS use this function when users ask for:
- Movie recommendations (e.g., "good action movies", "best comedies")
- Movies by genre (e.g., "sci-fi movies", "horror films")
- Movies by director or actor (e.g., "Christopher Nolan movies", "Tom Hanks films")
- Specific movie information (e.g., "tell me about Inception")
- Movie lists or suggestions
- Any query requiring actual movie data
DO NOT use this function when user ask for
- knowledge about previous aquired date
- more questions about previous conversation"""
# Function definition for LLM function calling
MOVIE_SEARCH_FUNCTION = {
    "name": "search_movie_database",
    "description": """MANDATORY: Search the comprehensive movie database to find movies with detailed information including title, year, director, cast, genre, plot, and ratings.
This function provides complete movie details from the database.""",
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Search query for movies (e.g., 'action movies', 'Christopher Nolan', 'sci-fi 2020', 'best comedies')"
            },
            "num_results": {
                "type": "integer",
                "description": "Number of movies to retrieve (default: 5, max: 10)",
                "default": 5,
                "minimum": 1,
                "maximum": 10
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
        llm_model: str = "mistral-large-latest",
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
        if self.llm_provider != "mistral":
            raise ValueError("Only Mistral provider is supported")

        api_key = os.environ.get("MISTRAL_API_KEY")
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
            print(messages)
            return self._call_mistral_with_functions(messages)
        except Exception as e:
            logger.error(f"Error calling Mistral LLM: {e}")
            return {
                "content": f"I apologize, but I encountered an error: {str(e)}",
                "function_calls": []
            }

    def _call_mistral_with_functions(self, messages: List[Dict[str, str]]) -> Dict[str, Any]:
        """Call Mistral API with function calling."""
        print("Messages-------------------------------------------------")
        print(messages)
        print("Messages End-------------------------------------------------")
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

            # Check if this looks like a movie query to force function calling
            user_message = messages[-1]["content"].lower() if messages else ""
            movie_keywords = [
                "movie", "film", "recommend", "genre", "director", "actor", "actress",
                "action", "comedy", "drama", "horror", "sci-fi", "thriller", "romance",
                "best", "good", "watch", "cinema", "plot", "cast", "year"
            ]

            # Force function calling for movie-related queries
            is_movie_query = any(keyword in user_message for keyword in movie_keywords)
            tool_choice = "auto"#"any" if is_movie_query else "auto"

            logger.info(f"Query: '{user_message[:50]}...' | Movie query: {is_movie_query} | Tool choice: {tool_choice}")

            response = client.chat.complete(
                model=self.llm_model,
                messages=messages,
                tools=tools,
                tool_choice=tool_choice,
                temperature=0.3  # Lower temperature for more consistent function calling
            )

            message = response.choices[0].message
            print("Response-------------------------------------------------")
            print(response)
            print("Response End-------------------------------------------------")

            result = {
                "content": message.content or "",
                "function_calls": [],
                "original_tool_calls": []  # Store original tool call objects
            }

            if message.tool_calls:
                for tool_call in message.tool_calls:
                    if tool_call.function:
                        result["function_calls"].append({
                            "name": tool_call.function.name,
                            "arguments": json.loads(tool_call.function.arguments)
                        })
                        # Store the original tool call object for proper message formatting
                        result["original_tool_calls"].append(tool_call)

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
            "content": """You are a movie expert assistant with access to a comprehensive movie database. Your primary job is to help users find movies and provide movie information.

            CRITICAL RULES:
            1. ALWAYS use the search_movie_database function for ANY movie-related query
            2. NEVER try to answer movie questions from memory - always search the database first
            3. Use the function for: recommendations, movie lists, genre searches, director/actor queries, specific movie info
            3. If user ask movies in messages, use the data from the messages.

            Examples that REQUIRE the function:
            - "What are good action movies?" → search_movie_database(query="action movies")
            - "Movies by Christopher Nolan" → search_movie_database(query="Christopher Nolan")
            - "Best sci-fi films" → search_movie_database(query="sci-fi movies")
            - "Tell me about Inception" → search_movie_database(query="Inception")

            Only respond without the function for: greetings, thanks, general conversation unrelated to movies.

            After getting search results, provide detailed, helpful responses based on the actual data from the database."""
        }]
        """"""
        # Add conversation history
        for turn in self.conversation_history:
            messages.append({"role": "user", "content": turn["user_message"]})
            messages.append({"role": "assistant", "content": turn["assistant_message"]})

        # Add current user query
        messages.append({"role": "user", "content": user_query})

        return messages

    def _is_movie_query(self, query: str) -> bool:
        """Determine if a query is movie-related."""
        query_lower = query.lower()
        movie_keywords = [
            "movie", "film", "recommend", "genre", "director", "actor", "actress",
            "action", "comedy", "drama", "horror", "sci-fi", "thriller", "romance",
            "best", "good", "watch", "cinema", "plot", "cast", "year", "netflix",
            "imdb", "oscar", "award", "blockbuster", "sequel", "franchise"
        ]
        return any(keyword in query_lower for keyword in movie_keywords)

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

            # Check if this was a movie query that should have triggered function calling
            is_movie_query = self._is_movie_query(user_query)
            # Execute any function calls
            function_results = []
            if llm_response["function_calls"]:
                function_results = self._execute_function_calls(llm_response["function_calls"])

                # If functions were called, make a second LLM call with results
                if function_results:
                    # First, add the assistant's response with tool calls to messages
                    assistant_message = {
                        "role": "assistant",
                        "content": llm_response["content"] or ""
                    }

                    # Add tool calls to the assistant message if we have the original tool calls
                    if "original_tool_calls" in llm_response and llm_response["original_tool_calls"]:
                        assistant_message["tool_calls"] = llm_response["original_tool_calls"]

                    messages.append(assistant_message)
                    
                    print("\ntool_calls--------------------------------------------------------")
                    print(assistant_message["tool_calls"])
                    print("tool_calls End--------------------------------------------------------\n")
                    # Then add function results to messages
                    for func_result in function_results:
                        function_message = {
                            "role": "tool",
                            "id": assistant_message['tool_calls'][-1].id,
                            "tool_call_id": assistant_message['tool_calls'][-1].id,
                            "name": func_result["function_name"],
                            "content": json.dumps(func_result["result"][:3])  # Limit to top 3 results
                        }
                        
                        messages.append(function_message)
                    
                    print("\nMessages-------------------------------------------------")
                    print(messages)
                    print("Messages End-------------------------------------------------\n")

                    final_response = self._call_llm_with_functions(messages)
                    response_text = final_response["content"]
                else:
                    response_text = llm_response["content"]
            else:
                # If no function was called but this was a movie query, force a search
                if is_movie_query and False:
                    logger.warning(f"Movie query detected but no function called. Forcing search for: {user_query}")

                    # Force a database search
                    forced_results = self._search_movie_database(user_query, 5)
                    function_results = [{
                        "function_name": "search_movie_database",
                        "arguments": {"query": user_query, "num_results": 5},
                        "result": forced_results
                    }]

                    # Add forced function result to messages
                    function_message = {
                        "role": "function",
                        "name": "search_movie_database",
                        "content": json.dumps(forced_results[:3])
                    }
                    messages.append(function_message)

                    # Get response with the forced search results
                    final_response = self._call_llm_with_functions(messages)
                    response_text = final_response["content"]

                    # Update the function calls list to reflect what actually happened
                    llm_response["function_calls"] = [{
                        "name": "search_movie_database",
                        "arguments": {"query": user_query, "num_results": 5}
                    }]
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
        llm_model="mistral-large-latest"
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
