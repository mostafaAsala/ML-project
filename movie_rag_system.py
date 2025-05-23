"""
Movie Retrieval-Augmented Generation (RAG) System

This module implements a RAG system for movie-related question answering that combines:
1. Vector-based retrieval using the MovieVectorDB
2. LLM-based generation using the LLMSummarizer
3. Context processing and prompt engineering for accurate responses

The system can answer questions about movies, provide recommendations,
and generate detailed responses based on the movie database.
"""

import os
import json
import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import pandas as pd

from movie_vector_db import MovieVectorDB
from llm_summarizer import LLMSummarizer

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MovieRAGSystem:
    """
    A Retrieval-Augmented Generation system for movie-related question answering.
    
    This class combines vector-based retrieval with LLM generation to provide
    accurate and contextual answers about movies.
    """
    
    def __init__(
        self,
        vector_db_path: Optional[str] = None,
        llm_provider: str = "openai",
        llm_model: str = "gpt-3.5-turbo",
        embedding_model: str = "all-MiniLM-L6-v2",
        max_context_length: int = 4000,
        top_k_retrieval: int = 5
    ):
        """
        Initialize the RAG system.
        
        Args:
            vector_db_path: Path to the vector database
            llm_provider: LLM provider ('openai', 'mistral', 'anthropic', etc.)
            llm_model: Specific model name
            embedding_model: Sentence transformer model for embeddings
            max_context_length: Maximum context length for LLM
            top_k_retrieval: Number of documents to retrieve
        """
        self.vector_db_path = vector_db_path
        self.llm_provider = llm_provider
        self.llm_model = llm_model
        self.max_context_length = max_context_length
        self.top_k_retrieval = top_k_retrieval
        
        # Initialize components
        self.vector_db = MovieVectorDB(model_name=embedding_model)
        self.llm_summarizer = LLMSummarizer(
            model_name=llm_model,
            provider=llm_provider,
            max_tokens=1000,
            temperature=0.7
        )
        
        # Load vector database if path provided
        if vector_db_path:
            self.load_vector_db(vector_db_path)
        
        # Conversation history
        self.conversation_history = []
        
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
    
    def retrieve_context(self, query: str, k: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Retrieve relevant movie information based on the query.
        
        Args:
            query: User query
            k: Number of results to retrieve (defaults to top_k_retrieval)
            
        Returns:
            List of relevant movie documents
        """
        if self.vector_db.index is None:
            raise ValueError("Vector database not loaded. Call load_vector_db() or setup_vector_db() first.")
        
        k = k or self.top_k_retrieval
        
        try:
            results = self.vector_db.search(query, k=k)
            logger.info(f"Retrieved {len(results)} relevant documents for query: '{query[:50]}...'")
            return results
        except Exception as e:
            logger.error(f"Error retrieving context: {e}")
            return []
    
    def format_context(self, retrieved_docs: List[Dict[str, Any]]) -> str:
        """
        Format retrieved documents into a context string for the LLM.
        
        Args:
            retrieved_docs: List of retrieved movie documents
            
        Returns:
            Formatted context string
        """
        if not retrieved_docs:
            return "No relevant movie information found."
        
        context_parts = []
        for i, doc in enumerate(retrieved_docs, 1):
            # Extract key information
            title = doc.get('title', 'Unknown Title')
            year = doc.get('year', 'Unknown Year')
            director = doc.get('director', 'Unknown Director')
            genre = doc.get('genre', 'Unknown Genre')
            plot = doc.get('plot', 'No plot available')
            cast = doc.get('cast', 'Unknown Cast')
            similarity_score = doc.get('similarity_score', 0)
            
            # Format the document
            doc_text = f"""
Movie {i}: {title} ({year})
Director: {director}
Genre: {genre}
Cast: {cast}
Plot: {plot[:500]}{'...' if len(str(plot)) > 500 else ''}
Relevance Score: {similarity_score:.3f}
"""
            context_parts.append(doc_text.strip())
        
        return "\n\n".join(context_parts)
    
    def create_prompt(self, query: str, context: str, conversation_history: List[Dict] = None) -> str:
        """
        Create a prompt for the LLM that includes the query, context, and conversation history.
        
        Args:
            query: User query
            context: Retrieved context
            conversation_history: Previous conversation turns
            
        Returns:
            Formatted prompt string
        """
        # Base system prompt
        system_prompt = """You are a knowledgeable movie expert assistant. You help users find information about movies, provide recommendations, and answer questions based on the movie database.

Instructions:
1. Use the provided movie context to answer questions accurately
2. If the context doesn't contain enough information, say so clearly
3. Provide specific details when available (titles, years, directors, cast, etc.)
4. For recommendations, explain why you're suggesting specific movies
5. Be conversational and helpful
6. If asked about movies not in the context, acknowledge the limitation

Movie Database Context:
{context}

"""
        
        # Add conversation history if available
        history_text = ""
        if conversation_history:
            history_text = "\nPrevious Conversation:\n"
            for turn in conversation_history[-3:]:  # Last 3 turns
                history_text += f"User: {turn['query']}\n"
                history_text += f"Assistant: {turn['response'][:200]}...\n\n"
        
        # Combine everything
        full_prompt = system_prompt.format(context=context) + history_text + f"\nUser Question: {query}\n\nAssistant:"
        
        # Truncate if too long
        if len(full_prompt) > self.max_context_length:
            # Truncate context while keeping system prompt and query
            available_length = self.max_context_length - len(system_prompt) - len(f"\nUser Question: {query}\n\nAssistant:") - 200
            truncated_context = context[:available_length] + "...[truncated]"
            full_prompt = system_prompt.format(context=truncated_context) + f"\nUser Question: {query}\n\nAssistant:"
        
        return full_prompt
    
    def generate_response(self, prompt: str) -> str:
        """
        Generate a response using the LLM.
        
        Args:
            prompt: The formatted prompt
            
        Returns:
            Generated response
        """
        try:
            response = self.llm_summarizer._call_llm_api(prompt)
            return response.strip()
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return f"I apologize, but I encountered an error while generating a response: {str(e)}"
    
    def ask(self, query: str, include_context: bool = True, k: Optional[int] = None) -> Dict[str, Any]:
        """
        Ask a question and get a response from the RAG system.
        
        Args:
            query: User question
            include_context: Whether to include retrieved context in response
            k: Number of documents to retrieve
            
        Returns:
            Dictionary containing the response and metadata
        """
        start_time = datetime.now()
        
        try:
            # Step 1: Retrieve relevant context
            retrieved_docs = self.retrieve_context(query, k=k)
            
            # Step 2: Format context
            context = self.format_context(retrieved_docs)
            
            # Step 3: Create prompt
            prompt = self.create_prompt(query, context, self.conversation_history)
            
            # Step 4: Generate response
            response = self.generate_response(prompt)
            
            # Step 5: Prepare result
            result = {
                'query': query,
                'response': response,
                'retrieved_docs': retrieved_docs if include_context else [],
                'num_retrieved': len(retrieved_docs),
                'timestamp': start_time.isoformat(),
                'processing_time': (datetime.now() - start_time).total_seconds()
            }
            
            # Step 6: Update conversation history
            self.conversation_history.append({
                'query': query,
                'response': response,
                'timestamp': start_time.isoformat()
            })
            
            # Keep only last 10 conversations
            if len(self.conversation_history) > 10:
                self.conversation_history = self.conversation_history[-10:]
            
            logger.info(f"Successfully processed query in {result['processing_time']:.2f} seconds")
            return result
            
        except Exception as e:
            logger.error(f"Error processing query '{query}': {e}")
            return {
                'query': query,
                'response': f"I apologize, but I encountered an error while processing your question: {str(e)}",
                'retrieved_docs': [],
                'num_retrieved': 0,
                'timestamp': start_time.isoformat(),
                'processing_time': (datetime.now() - start_time).total_seconds(),
                'error': str(e)
            }
    
    def get_conversation_history(self) -> List[Dict]:
        """Get the conversation history."""
        return self.conversation_history.copy()
    
    def clear_conversation_history(self):
        """Clear the conversation history."""
        self.conversation_history = []
        logger.info("Conversation history cleared")
    
    def save_conversation(self, file_path: str):
        """
        Save conversation history to a JSON file.
        
        Args:
            file_path: Path to save the conversation
        """
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(self.conversation_history, f, indent=2, ensure_ascii=False)
            logger.info(f"Conversation saved to {file_path}")
        except Exception as e:
            logger.error(f"Error saving conversation: {e}")
    
    def load_conversation(self, file_path: str):
        """
        Load conversation history from a JSON file.
        
        Args:
            file_path: Path to load the conversation from
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                self.conversation_history = json.load(f)
            logger.info(f"Conversation loaded from {file_path}")
        except Exception as e:
            logger.error(f"Error loading conversation: {e}")


# Example usage and testing functions
if __name__ == "__main__":
    # Example usage
    print("Initializing Movie RAG System...")
    
    # Initialize the RAG system
    rag_system = MovieRAGSystem(
        llm_provider="mistral",  # Change to your preferred provider
        llm_model="mistral-small",
        top_k_retrieval=3
    )
    
    # Example questions
    example_questions = [
        "What are some good sci-fi movies?",
        "Tell me about movies directed by Christopher Nolan",
        "What movies are similar to The Matrix?",
        "Can you recommend some romantic comedies?",
        "What are the best movies from the 1990s?"
    ]
    
    print("RAG System initialized. Ready for questions!")
    print("Note: Make sure to set up the vector database first using setup_vector_db() method.")
