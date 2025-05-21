"""
Genre Vector Predictor

This module implements a genre prediction system that uses vector similarity
to predict movie genres. It leverages vector embeddings to find similar movies
and extract genre information from them.

The GenreVectorPredictor class can:
1. Use vector similarity to find movies with similar content
2. Extract genres from similar movies to predict genres for new movies
3. Weight genres based on similarity scores
4. Evaluate prediction performance
"""

import os
import json
import pickle
import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Any
from collections import Counter
from sklearn.metrics import f1_score, hamming_loss

# Import existing components
from movie_vector_db import MovieVectorDB


class GenreVectorPredictor:
    """
    A class that predicts movie genres using vector similarity.

    This class uses vector embeddings to find similar movies and extracts
    genre information from them, weighted by similarity scores.
    """

    def __init__(self, vector_db_path: Optional[str] = None,
                 vector_model: str = "all-MiniLM-L6-v2",
                 similarity_threshold: float = 0.7,
                 top_k: int = 10):
        """
        Initialize the GenreVectorPredictor.

        Args:
            vector_db_path: Path to the saved vector database
            vector_model: Name of the sentence-transformer model to use
            similarity_threshold: Minimum similarity score to consider
            top_k: Number of similar movies to consider
        """
        self.vector_db_path = vector_db_path or "saved_models/vector_db"
        self.vector_model = vector_model
        self.similarity_threshold = similarity_threshold
        self.top_k = top_k

        # Initialize components
        self.vector_db = None

        # Load vector database if path is provided
        if vector_db_path:
            self.load_vector_db(vector_db_path)

    def load_vector_db(self, path: Optional[str] = None) -> bool:
        """
        Load a vector database from disk.

        Args:
            path: Path to the vector database

        Returns:
            True if loaded successfully, False otherwise
        """
        path = path or self.vector_db_path

        # Initialize the vector database
        self.vector_db = MovieVectorDB(model_name=self.vector_model)

        # Load the database
        success = self.vector_db.load(path)

        if success:
            print(f"Vector database loaded from {path}")
        else:
            print(f"Failed to load vector database from {path}")

        return success



    def predict_genre_vector(self, query: str, threshold: Optional[float] = None,
                            k: Optional[int] = None) -> List[str]:
        """
        Predict genres using vector similarity.

        Args:
            query: Text query or movie description
            threshold: Minimum similarity score to consider
            k: Number of similar movies to consider

        Returns:
            List of predicted genres
        """
        if self.vector_db is None:
            raise ValueError("Vector database not loaded. Call load_vector_db() first.")

        # Use default values if not provided
        threshold = threshold or self.similarity_threshold
        k = k or self.top_k

        # Search for similar movies
        similar_movies = self.vector_db.search(query, k=k)

        # Filter by similarity threshold
        similar_movies = [movie for movie in similar_movies
                         if movie.get('similarity_score', 0) >= threshold]

        if not similar_movies:
            return []

        # Extract genres from similar movies
        genre_counts = Counter()

        for movie in similar_movies:
            # Handle different genre formats
            if 'genre' in movie:
                # Handle string format (comma-separated)
                if isinstance(movie['genre'], str):
                    genres = [g.strip() for g in movie['genre'].split(',')]
                    genre_counts.update(genres)
                # Handle list format
                elif isinstance(movie['genre'], list):
                    genre_counts.update(movie['genre'])

            # Handle genre_list format (used in some datasets)
            elif 'genre_list' in movie:
                if isinstance(movie['genre_list'], list):
                    genre_counts.update(movie['genre_list'])
                elif isinstance(movie['genre_list'], str):
                    # Handle string representation of list
                    if movie['genre_list'].startswith('[') and movie['genre_list'].endswith(']'):
                        import ast
                        try:
                            genres = ast.literal_eval(movie['genre_list'])
                            genre_counts.update(genres)
                        except:
                            pass

        # Get the most common genres
        # Weight by frequency and similarity score
        weighted_genres = {}
        for movie in similar_movies:
            similarity = movie.get('similarity_score', 0)

            # Extract genres
            movie_genres = []
            if 'genre' in movie:
                if isinstance(movie['genre'], str):
                    movie_genres = [g.strip() for g in movie['genre'].split(',')]
                elif isinstance(movie['genre'], list):
                    movie_genres = movie['genre']
            elif 'genre_list' in movie:
                if isinstance(movie['genre_list'], list):
                    movie_genres = movie['genre_list']
                elif isinstance(movie['genre_list'], str) and movie['genre_list'].startswith('['):
                    import ast
                    try:
                        movie_genres = ast.literal_eval(movie['genre_list'])
                    except:
                        pass

            # Update weighted scores
            for genre in movie_genres:
                if genre in weighted_genres:
                    weighted_genres[genre] += similarity
                else:
                    weighted_genres[genre] = similarity

        # Sort genres by weighted score
        sorted_genres = sorted(weighted_genres.items(), key=lambda x: x[1], reverse=True)

        # Return top genres (those with at least half the score of the top genre)
        if sorted_genres:
            top_score = sorted_genres[0][1]
            threshold_score = top_score * 0.5
            return [genre for genre, score in sorted_genres if score >= threshold_score]

        return []



    def evaluate(self, test_data: pd.DataFrame, plot_col: str = 'plot',
                genre_col: str = 'genre_list') -> Dict[str, float]:
        """
        Evaluate the genre prediction performance.

        Args:
            test_data: DataFrame containing test data
            plot_col: Column name for plot text
            genre_col: Column name for true genres

        Returns:
            Dictionary with evaluation metrics
        """
        # Prepare for evaluation
        y_true = []
        y_pred = []

        # Get all unique genres
        all_genres = set()
        for genres in test_data[genre_col]:
            if isinstance(genres, list):
                all_genres.update(genres)
            elif isinstance(genres, str) and genres.startswith('['):
                import ast
                try:
                    genre_list = ast.literal_eval(genres)
                    all_genres.update(genre_list)
                except:
                    pass

        all_genres = sorted(list(all_genres))

        # Make predictions for each movie
        for _, row in test_data.iterrows():
            # Get true genres
            true_genres = []
            if isinstance(row[genre_col], list):
                true_genres = row[genre_col]
            elif isinstance(row[genre_col], str) and row[genre_col].startswith('['):
                import ast
                try:
                    true_genres = ast.literal_eval(row[genre_col])
                except:
                    pass

            # Make vector-based prediction
            pred_genres = self.predict_genre_vector(row[plot_col])

            # Convert to binary vectors
            y_true.append([1 if g in true_genres else 0 for g in all_genres])
            y_pred.append([1 if g in pred_genres else 0 for g in all_genres])

        # Calculate metrics
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)

        metrics = {
            'f1_micro': f1_score(y_true, y_pred, average='micro'),
            'f1_macro': f1_score(y_true, y_pred, average='macro'),
            'f1_weighted': f1_score(y_true, y_pred, average='weighted'),
            'hamming_loss': hamming_loss(y_true, y_pred)
        }

        return metrics

    def save(self, path: str = "saved_models/genre_vector_predictor") -> Dict[str, Any]:
        """
        Save the predictor configuration.

        Args:
            path: Path to save the configuration

        Returns:
            Dictionary with save information
        """
        # Create directory if it doesn't exist
        os.makedirs(path, exist_ok=True)

        # Save configuration
        config = {
            'vector_db_path': self.vector_db_path,
            'vector_model': self.vector_model,
            'similarity_threshold': self.similarity_threshold,
            'top_k': self.top_k
        }

        config_path = os.path.join(path, "config.json")
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=4)

        print(f"Predictor configuration saved to {config_path}")

        return {
            'path': path,
            'config_path': config_path,
            'config': config
        }

    @classmethod
    def load_config(cls, path: str = "saved_models/genre_vector_predictor") -> 'GenreVectorPredictor':
        """
        Load a predictor from a saved configuration.

        Args:
            path: Path to the saved configuration

        Returns:
            GenreVectorPredictor instance
        """
        config_path = os.path.join(path, "config.json")

        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found at {config_path}")

        with open(config_path, 'r') as f:
            config = json.load(f)

        # Create a new instance with the loaded configuration
        predictor = cls(
            vector_db_path=config.get('vector_db_path'),
            vector_model=config.get('vector_model'),
            similarity_threshold=config.get('similarity_threshold'),
            top_k=config.get('top_k')
        )

        return predictor
