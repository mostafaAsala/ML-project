"""
Movie RAG System Streamlit App

An interactive web interface for the Movie RAG System that allows users to:
1. Ask questions about movies
2. Get recommendations
3. View conversation history
4. Configure system settings
5. See retrieved context and sources
"""

import streamlit as st
import os
import json
import pandas as pd
from datetime import datetime
from movie_rag_system import MovieRAGSystem

# Page configuration
st.set_page_config(
    page_title="Movie RAG System",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
.main-header {
    font-size: 2.5rem;
    color: #1f77b4;
    text-align: center;
    margin-bottom: 2rem;
}
.chat-message {
    padding: 1rem;
    border-radius: 0.5rem;
    margin: 0.5rem 0;
}
.user-message {
    background-color: #e3f2fd;
    color: #000000;
    border-left: 4px solid #2196f3;
}
.assistant-message {
    background-color: #f3e5f5;
    color: #000000;
    border-left: 4px solid #9c27b0;
}
.movie-card {
    border: 1px solid #ddd;
    border-radius: 8px;
    padding: 1rem;
    margin: 0.5rem 0;
    background-color: #f9f9f9;
}
.metric-card {
    background-color: #f0f2f6;
    padding: 1rem;
    border-radius: 0.5rem;
    text-align: center;
}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def initialize_rag_system():
    """Initialize and cache the RAG system."""
    try:
        # Get configuration from session state or use defaults
        llm_provider = st.session_state.get('llm_provider', 'mistral')
        llm_model = st.session_state.get('llm_model', 'mistral-small')
        top_k = st.session_state.get('top_k_retrieval', 5)
        
        rag_system = MovieRAGSystem(
            llm_provider=llm_provider,
            llm_model=llm_model,
            top_k_retrieval=top_k
        )
        
        # Try to load existing vector database
        vector_db_path = "saved_models/vector_db"
        if os.path.exists(os.path.join(vector_db_path, "faiss_index.bin")):
            success = rag_system.load_vector_db(vector_db_path)
            if success:
                return rag_system, "Vector database loaded successfully!"
            else:
                return None, "Failed to load vector database"
        else:
            return None, "Vector database not found. Please set it up first."
            
    except Exception as e:
        return None, f"Error initializing RAG system: {str(e)}"

def setup_sidebar():
    """Setup the sidebar with configuration options."""
    st.sidebar.title("🎬 Movie RAG System")
    st.sidebar.markdown("---")
    
    # System Configuration
    st.sidebar.subheader("⚙️ Configuration")
    
    # LLM Provider selection
    llm_provider = st.sidebar.selectbox(
        "LLM Provider",
        ["mistral", "openai", "anthropic", "huggingface"],
        index=0,
        key="llm_provider"
    )
    
    # Model selection based on provider
    model_options = {
        "mistral": ["mistral-small", "mistral-medium", "mistral-large"],
        "openai": ["gpt-3.5-turbo", "gpt-4"],
        "anthropic": ["claude-3-haiku", "claude-3-sonnet", "claude-3-opus"],
        "huggingface": ["microsoft/DialoGPT-medium", "facebook/blenderbot-400M-distill"]
    }
    
    llm_model = st.sidebar.selectbox(
        "Model",
        model_options.get(llm_provider, ["default"]),
        key="llm_model"
    )
    
    # Retrieval settings
    top_k_retrieval = st.sidebar.slider(
        "Number of movies to retrieve",
        min_value=1,
        max_value=10,
        value=5,
        key="top_k_retrieval"
    )
    
    st.sidebar.markdown("---")
    
    # API Key management
    st.sidebar.subheader("🔑 API Keys")
    
    if llm_provider == "openai":
        api_key = st.sidebar.text_input("OpenAI API Key", type="password")
        if api_key:
            os.environ["OPENAI_API_KEY"] = api_key
    elif llm_provider == "mistral":
        api_key = st.sidebar.text_input("Mistral API Key", type="password")
        if api_key:
            os.environ["MISTRAL_API_KEY"] = api_key
    elif llm_provider == "anthropic":
        api_key = st.sidebar.text_input("Anthropic API Key", type="password")
        if api_key:
            os.environ["ANTHROPIC_API_KEY"] = api_key
    
    st.sidebar.markdown("---")
    
    # Database setup
    st.sidebar.subheader("🗄️ Database Setup")
    
    if st.sidebar.button("Setup Vector Database"):
        setup_database()
    
    # Conversation management
    st.sidebar.subheader("💬 Conversation")
    
    col1, col2 = st.sidebar.columns(2)
    with col1:
        if st.button("Clear History"):
            if 'rag_system' in st.session_state:
                st.session_state.rag_system.clear_conversation_history()
            st.session_state.conversation_history = []
            st.rerun()
    
    with col2:
        if st.button("Save Chat"):
            save_conversation()

def setup_database():
    """Setup the vector database."""
    st.sidebar.info("Setting up vector database...")
    
    # File uploader for movie data
    uploaded_file = st.sidebar.file_uploader(
        "Upload movie data CSV",
        type=['csv'],
        help="Upload your movie dataset (CSV format)"
    )
    
    if uploaded_file is not None:
        try:
            # Save uploaded file temporarily
            temp_path = f"temp_{uploaded_file.name}"
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            # Initialize RAG system for setup
            rag_system = MovieRAGSystem()
            
            # Setup vector database
            with st.spinner("Setting up vector database... This may take a few minutes."):
                success = rag_system.setup_vector_db(temp_path, data_source="wiki")
            
            if success:
                st.sidebar.success("Vector database setup completed!")
                # Clear cache to reload with new database
                st.cache_resource.clear()
            else:
                st.sidebar.error("Failed to setup vector database")
            
            # Clean up temporary file
            os.remove(temp_path)
            
        except Exception as e:
            st.sidebar.error(f"Error setting up database: {str(e)}")

def save_conversation():
    """Save the current conversation."""
    if 'rag_system' in st.session_state:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"conversation_{timestamp}.json"
        st.session_state.rag_system.save_conversation(filename)
        st.sidebar.success(f"Conversation saved as {filename}")

def display_movie_card(movie_data):
    """Display a movie information card."""
    title = movie_data.get('title', 'Unknown Title')
    year = movie_data.get('year', 'Unknown Year')
    director = movie_data.get('director', 'Unknown Director')
    genre = movie_data.get('genre', 'Unknown Genre')
    plot = movie_data.get('plot', 'No plot available')
    cast = movie_data.get('cast', 'Unknown Cast')
    similarity_score = movie_data.get('similarity_score', 0)
    
    with st.container():
        st.markdown(f"""
        <div class="movie-card">
            <h4>{title} ({year})</h4>
            <p><strong>Director:</strong> {director}</p>
            <p><strong>Genre:</strong> {genre}</p>
            <p><strong>Cast:</strong> {cast}</p>
            <p><strong>Plot:</strong> {plot[:200]}{'...' if len(str(plot)) > 200 else ''}</p>
            <p><strong>Relevance Score:</strong> {similarity_score:.3f}</p>
        </div>
        """, unsafe_allow_html=True)

def main():
    """Main application function."""
    # Header
    st.markdown('<h1 class="main-header">🎬 Movie RAG System</h1>', unsafe_allow_html=True)
    st.markdown("Ask questions about movies, get recommendations, and explore our movie database!")
    
    # Setup sidebar
    setup_sidebar()
    
    # Initialize RAG system
    if 'rag_system' not in st.session_state:
        rag_system, message = initialize_rag_system()
        if rag_system:
            st.session_state.rag_system = rag_system
            st.success(message)
        else:
            st.error(message)
            st.stop()
    
    # Initialize conversation history
    if 'conversation_history' not in st.session_state:
        st.session_state.conversation_history = []
    
    # Main interface
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("💬 Chat with the Movie Expert")
        
        # Chat input
        user_question = st.text_input(
            "Ask me anything about movies:",
            placeholder="e.g., What are some good sci-fi movies? or Tell me about Christopher Nolan films",
            key="user_input"
        )
        
        # Process question
        if user_question and st.button("Ask", type="primary"):
            with st.spinner("Thinking..."):
                result = st.session_state.rag_system.ask(user_question)
                
                # Add to conversation history
                st.session_state.conversation_history.append({
                    'question': user_question,
                    'answer': result['response'],
                    'retrieved_docs': result['retrieved_docs'],
                    'timestamp': result['timestamp'],
                    'processing_time': result['processing_time']
                })
        
        # Display conversation history
        st.subheader("📝 Conversation History")
        
        if st.session_state.conversation_history:
            for i, turn in enumerate(reversed(st.session_state.conversation_history)):
                with st.expander(f"Q: {turn['question'][:50]}...", expanded=(i == 0)):
                    st.markdown(f"""
                    <div class="chat-message user-message">
                        <strong>You:</strong> {turn['question']}
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.markdown(f"""
                    <div class="chat-message assistant-message">
                        <strong>Assistant:</strong> {turn['answer']}
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.caption(f"⏱️ {turn['processing_time']:.2f}s | 📚 {len(turn['retrieved_docs'])} sources")
        else:
            st.info("Start a conversation by asking a question about movies!")
    
    with col2:
        st.subheader("📊 System Status")
        
        # System metrics
        if 'rag_system' in st.session_state:
            rag_system = st.session_state.rag_system
            
            # Database info
            if rag_system.vector_db.movies_df is not None:
                num_movies = len(rag_system.vector_db.movies_df)
                st.metric("Movies in Database", num_movies)
            
            # Conversation stats
            num_conversations = len(st.session_state.conversation_history)
            st.metric("Questions Asked", num_conversations)
            
            # Current settings
            st.metric("Retrieval Count", rag_system.top_k_retrieval)
        
        # Recent sources
        st.subheader("📚 Recent Sources")
        
        if (st.session_state.conversation_history and 
            st.session_state.conversation_history[-1]['retrieved_docs']):
            
            recent_docs = st.session_state.conversation_history[-1]['retrieved_docs'][:3]
            
            for doc in recent_docs:
                with st.container():
                    title = doc.get('title', 'Unknown')
                    year = doc.get('year', 'Unknown')
                    score = doc.get('similarity_score', 0)
                    
                    st.markdown(f"""
                    <div style="border: 1px solid #ddd; padding: 0.5rem; margin: 0.25rem 0; border-radius: 4px;">
                        <strong>{title}</strong> ({year})<br>
                        <small>Relevance: {score:.3f}</small>
                    </div>
                    """, unsafe_allow_html=True)
        
        # Quick actions
        st.subheader("🚀 Quick Actions")
        
        quick_questions = [
            "What are some popular movies?",
            "Recommend a comedy movie",
            "Tell me about action movies",
            "What are classic films?",
            "Suggest a horror movie"
        ]
        
        for question in quick_questions:
            if st.button(question, key=f"quick_{question}"):
                st.session_state.user_input = question
                st.rerun()

if __name__ == "__main__":
    main()
