# Movie RAG System

A comprehensive Retrieval-Augmented Generation (RAG) system for movie-related question answering that combines vector-based retrieval with large language model generation.

## 🎯 Overview

The Movie RAG System enables users to:
- Ask natural language questions about movies
- Get intelligent recommendations based on preferences
- Retrieve contextual information from a movie database
- Have conversational interactions with memory
- Access information through multiple interfaces (CLI, Streamlit web app)

## 🏗️ Architecture

The system consists of three main components:

1. **Vector Database** (`MovieVectorDB`): Stores and retrieves movie information using semantic embeddings
2. **LLM Integration** (`LLMSummarizer`): Generates responses using various language models
3. **RAG Orchestrator** (`MovieRAGSystem`): Combines retrieval and generation for coherent answers

```
User Query → Vector Retrieval → Context Formatting → LLM Generation → Response
```

## 📁 Files Structure

```
├── movie_rag_system.py          # Main RAG system implementation
├── rag_example.py               # Usage examples and demonstrations
├── rag_streamlit_app.py         # Interactive web interface
├── test_rag_system.py           # Comprehensive test suite
├── movie_vector_db.py           # Vector database (existing)
├── llm_summarizer.py            # LLM integration (existing)
└── README_RAG_System.md         # This documentation
```

## 🚀 Quick Start

### 1. Installation

Install the required dependencies:

```bash
pip install -r requirements.txt
```

### 2. Set up API Keys

Set your API key for your preferred LLM provider:

```bash
# For OpenAI
export OPENAI_API_KEY="your-openai-api-key"

# For Mistral AI
export MISTRAL_API_KEY="your-mistral-api-key"

# For Anthropic
export ANTHROPIC_API_KEY="your-anthropic-api-key"
```

### 3. Initialize the System

```python
from movie_rag_system import MovieRAGSystem

# Initialize with your preferred LLM
rag_system = MovieRAGSystem(
    llm_provider="mistral",
    llm_model="mistral-small",
    top_k_retrieval=5
)

# Set up vector database (first time only)
rag_system.setup_vector_db("your_movie_data.csv", data_source="wiki")

# Or load existing database
rag_system.load_vector_db("saved_models/vector_db")
```

### 4. Ask Questions

```python
# Ask a question
result = rag_system.ask("What are some good sci-fi movies?")
print(result['response'])

# Get recommendations
result = rag_system.ask("I like complex plots and mind-bending twists. What do you recommend?")
print(result['response'])
```

## 💻 Usage Examples

### Basic Question Answering

```python
from movie_rag_system import MovieRAGSystem

# Initialize system
rag_system = MovieRAGSystem(llm_provider="mistral", llm_model="mistral-small")
rag_system.load_vector_db("saved_models/vector_db")

# Ask questions
questions = [
    "What are some good action movies?",
    "Tell me about Christopher Nolan films",
    "What movies are similar to The Matrix?",
    "Can you recommend some horror movies from the 1980s?"
]

for question in questions:
    result = rag_system.ask(question)
    print(f"Q: {question}")
    print(f"A: {result['response']}")
    print(f"Sources: {result['num_retrieved']} movies")
    print("-" * 50)
```

### Conversational Interaction

```python
# The system maintains conversation history automatically
rag_system.ask("What are some good action movies?")
rag_system.ask("Which of those has the best special effects?")  # Refers to previous context
rag_system.ask("Tell me more about the plot")  # Continues the conversation

# View conversation history
history = rag_system.get_conversation_history()
print(f"Conversation has {len(history)} turns")
```

### Advanced Configuration

```python
# Custom configuration
rag_system = MovieRAGSystem(
    llm_provider="openai",
    llm_model="gpt-4",
    embedding_model="all-MiniLM-L6-v2",
    max_context_length=4000,
    top_k_retrieval=7
)

# Retrieve more context for complex questions
result = rag_system.ask("Compare different sci-fi subgenres", k=10)
```

## 🌐 Web Interface

Launch the Streamlit web application:

```bash
streamlit run rag_streamlit_app.py
```

Features:
- Interactive chat interface
- Real-time configuration
- Conversation history
- Source document display
- API key management
- Database setup tools

## 🧪 Testing

Run the comprehensive test suite:

```bash
python test_rag_system.py
```

Or run specific test categories:

```bash
python -m unittest test_rag_system.TestMovieRAGSystem
python -m unittest test_rag_system.TestRAGSystemPerformance
python -m unittest test_rag_system.TestRAGSystemErrorHandling
```

## 📊 Supported LLM Providers

| Provider | Models | API Key Required |
|----------|--------|------------------|
| OpenAI | gpt-3.5-turbo, gpt-4 | Yes |
| Mistral AI | mistral-small, mistral-medium, mistral-large | Yes |
| Anthropic | claude-3-haiku, claude-3-sonnet, claude-3-opus | Yes |
| Hugging Face | Various models | Yes (for API) |
| Local Models | llama2, mistral (via llama-cpp) | No |

## 🔧 Configuration Options

### RAG System Parameters

- `llm_provider`: LLM provider ("openai", "mistral", "anthropic", etc.)
- `llm_model`: Specific model name
- `embedding_model`: Sentence transformer model for embeddings
- `max_context_length`: Maximum context length for LLM
- `top_k_retrieval`: Number of documents to retrieve

### Vector Database Parameters

- `model_name`: Embedding model for vector creation
- `batch_size`: Batch size for embedding creation
- `index_type`: FAISS index type (currently flat L2)

## 📈 Performance Considerations

### Optimization Tips

1. **Embedding Model**: Use smaller models for faster retrieval
2. **Context Length**: Limit context to essential information
3. **Retrieval Count**: Balance between relevance and speed
4. **Caching**: Vector database is cached after first load
5. **Batch Processing**: Process multiple queries efficiently

### Typical Performance

- **Vector Retrieval**: ~50-100ms for 10k movies
- **LLM Generation**: 1-5 seconds (depends on provider)
- **Total Response Time**: 2-8 seconds per query

## 🛠️ Troubleshooting

### Common Issues

1. **Vector Database Not Found**
   ```python
   # Set up the database first
   rag_system.setup_vector_db("your_data.csv")
   ```

2. **API Key Errors**
   ```bash
   # Set the appropriate environment variable
   export MISTRAL_API_KEY="your-key"
   ```

3. **Memory Issues**
   ```python
   # Reduce batch size for embedding creation
   rag_system.vector_db.create_embeddings(batch_size=16)
   ```

4. **Slow Responses**
   ```python
   # Reduce retrieval count
   rag_system.top_k_retrieval = 3
   ```

### Error Handling

The system includes comprehensive error handling:
- Graceful degradation when APIs are unavailable
- Fallback responses for retrieval failures
- Automatic retry mechanisms
- Detailed error logging

## 🔮 Future Enhancements

- [ ] Support for multimodal inputs (images, videos)
- [ ] Advanced retrieval strategies (hybrid search)
- [ ] Fine-tuned embedding models for movies
- [ ] Real-time database updates
- [ ] Multi-language support
- [ ] Integration with external movie APIs
- [ ] Advanced conversation management
- [ ] Personalization and user preferences

## 📝 License

This project is part of the Movie ML Project and follows the same licensing terms.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## 📞 Support

For issues and questions:
1. Check the troubleshooting section
2. Run the test suite to identify issues
3. Review the example files for usage patterns
4. Check API provider documentation for key setup
