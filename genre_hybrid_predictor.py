"""
Genre Hybrid Predictor

This module implements a hybrid genre prediction system that combines vector similarity
with traditional machine learning models for improved genre prediction.

The GenreHybridPredictor class can:
1. Use vector similarity to find movies with similar content
2. Use ML models for traditional feature-based prediction
3. Combine both approaches with configurable weights
4. Evaluate prediction performance
"""

import os
import json
import numpy as np
import pandas as pd
from typing import List, Dict, Union, Optional, Any
from collections import Counter
from sklearn.metrics import f1_score, hamming_loss

# Import existing components
from movie_vector_db import MovieVectorDB
from genre_predictor import GenrePredictor
from genre_vector_predictor import GenreVectorPredictor


class GenreHybridPredictor:
    """
    A class that combines vector-based and ML-based approaches for genre prediction.
    
    This class integrates two complementary approaches:
    1. Vector similarity: Find similar movies and use their genres
    2. ML models: Use traditional ML models for genre prediction
    
    The combination of these approaches can lead to more accurate predictions.
    """
    
    def __init__(self, vector_predictor: Optional[GenreVectorPredictor] = None,
                 ml_predictor: Optional[GenrePredictor] = None,
                 vector_db_path: Optional[str] = None,
                 genre_predictor_path: Optional[str] = None,
                 vector_model: str = "all-MiniLM-L6-v2",
                 similarity_threshold: float = 0.7,
                 top_k: int = 10,
                 weight_vector: float = 0.5,
                 weight_ml: float = 0.5):
        """
        Initialize the GenreHybridPredictor.
        
        Args:
            vector_predictor: Existing GenreVectorPredictor instance
            ml_predictor: Existing GenrePredictor instance
            vector_db_path: Path to the saved vector database
            genre_predictor_path: Path to the saved genre predictor
            vector_model: Name of the sentence-transformer model to use
            similarity_threshold: Minimum similarity score to consider
            top_k: Number of similar movies to consider
            weight_vector: Weight for vector-based predictions (0-1)
            weight_ml: Weight for ML-based predictions (0-1)
        """
        self.vector_db_path = vector_db_path or "saved_models/vector_db"
        self.genre_predictor_path = genre_predictor_path or "saved_models"
        self.vector_model = vector_model
        self.similarity_threshold = similarity_threshold
        self.top_k = top_k
        self.weight_vector = weight_vector
        self.weight_ml = weight_ml
        
        # Initialize components
        self.vector_predictor = vector_predictor
        self.ml_predictor = ml_predictor
        
        # Create components if not provided
        if self.vector_predictor is None:
            self.vector_predictor = GenreVectorPredictor(
                vector_db_path=vector_db_path,
                vector_model=vector_model,
                similarity_threshold=similarity_threshold,
                top_k=top_k
            )
        
        if self.ml_predictor is None and genre_predictor_path:
            self.load_ml_predictor(genre_predictor_path)
    
    def load_ml_predictor(self, path: Optional[str] = None) -> bool:
        """
        Load a genre predictor from disk.
        
        Args:
            path: Path to the genre predictor
            
        Returns:
            True if loaded successfully, False otherwise
        """
        path = path or self.genre_predictor_path
        
        try:
            # Load the genre predictor
            self.ml_predictor = GenrePredictor.load(models_dir=path)
            print(f"Genre predictor loaded from {path}")
            return True
        except Exception as e:
            print(f"Failed to load genre predictor: {e}")
            return False
    
    def predict_hybrid(self, query: str, data: Union[Dict, pd.DataFrame]) -> List[str]:
        """
        Predict genres using both vector similarity and ML models.
        
        Args:
            query: Text query or movie description
            data: Movie data for ML prediction
            
        Returns:
            List of predicted genres
        """
        # Get predictions from vector-based method
        vector_genres = self.vector_predictor.predict_genre_vector(query)
        
        # Get predictions from ML-based method
        if self.ml_predictor is None:
            print("Warning: ML predictor not loaded. Using only vector-based prediction.")
            return vector_genres
        
        ml_genres = self.ml_predictor.predict(data)
        
        # Handle different return formats from ML predictor
        if isinstance(ml_genres, list) and len(ml_genres) > 0 and isinstance(ml_genres[0], list):
            ml_genres = ml_genres[0]
        
        # Combine predictions with weights
        genre_scores = {}
        
        # Add vector-based predictions
        for genre in vector_genres:
            genre_scores[genre] = self.weight_vector
        
        # Add ML-based predictions
        for genre in ml_genres:
            if genre in genre_scores:
                genre_scores[genre] += self.weight_ml
            else:
                genre_scores[genre] = self.weight_ml
        
        # Sort by score
        sorted_genres = sorted(genre_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Return genres with score above 0.5 (present in at least one method with sufficient weight)
        return [genre for genre, score in sorted_genres if score >= 0.5]
    
    def evaluate(self, test_data: pd.DataFrame, plot_col: str = 'plot', 
                genre_col: str = 'genre_list', method: str = 'hybrid') -> Dict[str, float]:
        """
        Evaluate the genre prediction performance.
        
        Args:
            test_data: DataFrame containing test data
            plot_col: Column name for plot text
            genre_col: Column name for true genres
            method: Prediction method ('vector', 'ml', or 'hybrid')
            
        Returns:
            Dictionary with evaluation metrics
        """
        if method not in ['vector', 'ml', 'hybrid']:
            raise ValueError("Method must be 'vector', 'ml', or 'hybrid'")
        
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
            
            # Make prediction based on method
            if method == 'vector':
                pred_genres = self.vector_predictor.predict_genre_vector(row[plot_col])
            elif method == 'ml':
                if self.ml_predictor is None:
                    raise ValueError("ML predictor not loaded. Call load_ml_predictor() first.")
                pred_genres = self.ml_predictor.predict(row.to_dict())
                if isinstance(pred_genres, list) and len(pred_genres) > 0 and isinstance(pred_genres[0], list):
                    pred_genres = pred_genres[0]
            else:  # hybrid
                pred_genres = self.predict_hybrid(row[plot_col], row.to_dict())
            
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
    
    def save(self, path: str = "saved_models/genre_hybrid_predictor") -> Dict[str, Any]:
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
            'genre_predictor_path': self.genre_predictor_path,
            'vector_model': self.vector_model,
            'similarity_threshold': self.similarity_threshold,
            'top_k': self.top_k,
            'weight_vector': self.weight_vector,
            'weight_ml': self.weight_ml
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
    def load_config(cls, path: str = "saved_models/genre_hybrid_predictor") -> 'GenreHybridPredictor':
        """
        Load a predictor from a saved configuration.
        
        Args:
            path: Path to the saved configuration
            
        Returns:
            GenreHybridPredictor instance
        """
        config_path = os.path.join(path, "config.json")
        
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found at {config_path}")
        
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Create a new instance with the loaded configuration
        predictor = cls(
            vector_db_path=config.get('vector_db_path'),
            genre_predictor_path=config.get('genre_predictor_path'),
            vector_model=config.get('vector_model'),
            similarity_threshold=config.get('similarity_threshold'),
            top_k=config.get('top_k'),
            weight_vector=config.get('weight_vector'),
            weight_ml=config.get('weight_ml')
        )
        
        return predictor
