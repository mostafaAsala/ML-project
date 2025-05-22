# Genre Vector Predictor

This module implements a genre prediction system that uses vector similarity to predict movie genres. It leverages vector embeddings to find similar movies and extract genre information from them.

## Overview

The `GenreVectorPredictor` class uses vector similarity for movie genre prediction:

1. **Vector Similarity-Based Prediction**: Uses the `MovieVectorDB` to find movies with similar content and extracts genres from these similar movies.

2. **Weighted Genre Extraction**: Weights genres based on similarity scores to prioritize the most relevant genres.

## Features

- **Vector-Based Prediction**: Predict genres based on content similarity
- **Configurable Parameters**: Adjust similarity thresholds and number of similar movies
- **Evaluation Tools**: Built-in methods to evaluate prediction performance
- **Persistence**: Save and load predictor configurations

## Installation

No additional installation is required beyond the dependencies already used in the project:
- Python 3.6+
- NumPy
- Pandas
- scikit-learn
- FAISS
- sentence-transformers

## Usage

### Basic Usage

```python
from genre_vector_predictor import GenreVectorPredictor

# Create a new predictor
predictor = GenreVectorPredictor()

# Load the vector database
predictor.load_vector_db()

# Make predictions using vector similarity
genres = predictor.predict_genre_vector("A sci-fi movie about space exploration")
print(f"Predicted genres: {genres}")
```

### Customizing Prediction Parameters

```python
# Create a predictor with custom parameters
predictor = GenreVectorPredictor(
    similarity_threshold=0.8,  # Only consider movies with similarity score >= 0.8
    top_k=15                   # Consider top 15 similar movies
)
```

### Evaluating Prediction Performance

```python
import pandas as pd

# Load test data
test_data = pd.read_csv("movie_test_data.csv")

# Evaluate prediction performance
metrics = predictor.evaluate(test_data, plot_col='plot', genre_col='genre_list')

# Print results
print("Prediction metrics:")
for metric, value in metrics.items():
    print(f"  {metric}: {value:.4f}")
```

### Saving and Loading Configurations

```python
# Save the predictor configuration
predictor.save("saved_models/my_genre_predictor")

# Load a saved configuration
loaded_predictor = GenreVectorPredictor.load_config("saved_models/my_genre_predictor")
```

## Example Script

The `genre_vector_predictor_example.py` script provides a complete example of how to use the `GenreVectorPredictor` class:

```bash
# Run prediction example
python genre_vector_predictor_example.py --predict

# Run evaluation example
python genre_vector_predictor_example.py --evaluate --sample 100

# Create a new vector database
python genre_vector_predictor_example.py --create-db --data movie_data.csv

# Run all examples
python genre_vector_predictor_example.py
```

## How It Works

### Vector-Based Prediction

1. The movie description is encoded into a vector embedding using a sentence transformer model
2. The vector database is searched to find the most similar movies
3. Genres from similar movies are extracted and weighted by similarity score
4. The most common/relevant genres are returned as predictions

## Performance

Vector-based genre prediction typically achieves good performance, especially for common genres and when the vector database contains a diverse set of movies. Here's a typical performance example:

| Metric | Score |
|--------|-------|
| F1-Micro | 0.75 |
| F1-Macro | 0.62 |
| Hamming Loss | 0.15 |

## Integration with Other Components

The `GenreVectorPredictor` is designed to work seamlessly with other components in the project:

- **MovieVectorDB**: Provides vector embeddings and similarity search
- **GenreHybridPredictor**: Can use this class as a component in hybrid prediction

## Future Improvements

Potential enhancements for future versions:

1. **Genre-Specific Thresholds**: Use different thresholds for different genres
2. **Improved Weighting**: Develop more sophisticated weighting algorithms for genre extraction
3. **Fine-Tuning**: Allow fine-tuning of the vector embedding model specifically for genre prediction
4. **Contextual Embeddings**: Incorporate more contextual information in the vector embeddings
