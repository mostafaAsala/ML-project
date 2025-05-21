# Movie Vector Database and Search

This component implements a vector database for movies with semantic search capabilities. It allows users to search for movies using natural language queries, finding movies that are semantically similar to the query rather than just matching keywords.

## Features

- **Vector Embeddings**: Convert movie data (titles, plots, genres, etc.) into vector embeddings using sentence transformers
- **Efficient Search**: Use FAISS (Facebook AI Similarity Search) for fast and efficient similarity search
- **Flexible Data Sources**: Support for different movie data sources (Wiki Movie Plots, TMDB)
- **Persistence**: Save and load the vector database to/from disk
- **Interactive Search**: Simple command-line interface for interactive movie searches

## Requirements

The implementation requires the following dependencies:

- sentence-transformers: For creating semantic embeddings
- faiss-cpu: For efficient similarity search
- pandas: For data manipulation
- numpy: For numerical operations
- tqdm: For progress bars

These dependencies are included in the project's requirements.txt file.

## Usage

### Creating a Vector Database

To create a new vector database from movie data:

```python
from movie_vector_db import MovieVectorDB

# Initialize the vector database
db = MovieVectorDB()

# Load and preprocess the data
db.load_data("wiki_movie_plots_cleaned.csv", data_source="wiki")
db.preprocess_data()

# Create embeddings and build the index
db.create_embeddings()
db.build_index()

# Save the database
db.save()
```

### Loading an Existing Database

To load a previously saved vector database:

```python
from movie_vector_db import MovieVectorDB

# Initialize the vector database
db = MovieVectorDB()

# Load the database
db.load()
```

### Searching for Movies

To search for movies using natural language queries:

```python
# Search for movies about space exploration
results = db.search("space exploration adventure", k=5)

# Search for romantic comedies set in New York
results = db.search("romantic comedy in New York", k=5)

# Search for action movies with car chases
results = db.search("action movie with exciting car chases", k=5)
```

## Example Script

The `movie_vector_search_example.py` script provides a complete example of how to use the vector database:

```bash
# Create a new vector database
python movie_vector_search_example.py --create --data wiki_movie_plots_cleaned.csv --source wiki

# Load an existing database and run interactive search
python movie_vector_search_example.py
```

## How It Works

1. **Data Loading**: Movie data is loaded from CSV files containing movie information.
2. **Preprocessing**: The data is preprocessed to create text representations suitable for embedding.
3. **Embedding Creation**: The text representations are converted into vector embeddings using a pre-trained sentence transformer model.
4. **Index Building**: The embeddings are added to a FAISS index for efficient similarity search.
5. **Searching**: User queries are converted into embeddings and compared to the movie embeddings to find the most similar movies.

## Customization

You can customize the vector database by:

- Using different embedding models (e.g., "all-mpnet-base-v2" for higher quality but slower embeddings)
- Adjusting the text representation to emphasize different aspects of movies
- Using different similarity metrics (L2, cosine, inner product)
- Implementing more advanced FAISS indexes for larger datasets

## Performance Considerations

- The initial creation of embeddings can be time-consuming for large datasets
- Once the index is built, searches are very fast
- Memory usage depends on the size of the dataset and the dimensionality of the embeddings
- For very large datasets, consider using a more memory-efficient FAISS index type

## Future Improvements

Potential improvements to the vector database include:

- Hybrid search combining vector similarity with keyword filtering
- Clustering movies by similarity for better exploration
- Implementing user feedback to improve search results
- Adding support for more data sources
- Creating a web interface for easier interaction
