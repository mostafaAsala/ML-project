"""
Unit tests for the GenreVectorPredictor class.
"""

import os
import shutil
import unittest
import tempfile
import pandas as pd
import numpy as np
from unittest.mock import MagicMock, patch

from genre_vector_predictor import GenreVectorPredictor
from movie_vector_db import MovieVectorDB
from genre_predictor import GenrePredictor


class TestGenreVectorPredictor(unittest.TestCase):
    """Test cases for the GenreVectorPredictor class."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test data and mocks."""
        # Create a temporary directory for test data
        cls.test_dir = tempfile.mkdtemp()
        
        # Create test data
        cls.test_data = pd.DataFrame({
            'title': ['Movie 1', 'Movie 2', 'Movie 3', 'Movie 4', 'Movie 5'],
            'plot': [
                'A sci-fi adventure in space with aliens and advanced technology.',
                'A romantic comedy about two people falling in love in Paris.',
                'A crime drama about a mafia family in New York.',
                'A horror movie with supernatural elements and a haunted house.',
                'An action movie with car chases and explosions.'
            ],
            'plot_lemmatized': [
                'sci-fi adventure space alien advanced technology',
                'romantic comedy two people fall love paris',
                'crime drama mafia family new york',
                'horror movie supernatural element haunted house',
                'action movie car chase explosion'
            ],
            'genre': [
                'Sci-Fi, Adventure',
                'Romance, Comedy',
                'Crime, Drama',
                'Horror, Thriller',
                'Action'
            ],
            'genre_list': [
                ['Sci-Fi', 'Adventure'],
                ['Romance', 'Comedy'],
                ['Crime', 'Drama'],
                ['Horror', 'Thriller'],
                ['Action']
            ],
            'Origin/Ethnicity': [
                'American',
                'French',
                'American',
                'British',
                'American'
            ]
        })
        
        # Save test data to CSV
        cls.test_csv = os.path.join(cls.test_dir, 'test_movies.csv')
        cls.test_data.to_csv(cls.test_csv, index=False)
        
        # Create a mock vector database
        cls.mock_vector_db = MagicMock(spec=MovieVectorDB)
        cls.mock_vector_db.search.return_value = [
            {
                'title': 'Similar Movie 1',
                'genre': 'Sci-Fi, Adventure',
                'similarity_score': 0.9
            },
            {
                'title': 'Similar Movie 2',
                'genre': 'Sci-Fi, Action',
                'similarity_score': 0.8
            },
            {
                'title': 'Similar Movie 3',
                'genre': 'Adventure',
                'similarity_score': 0.7
            }
        ]
        
        # Create a mock genre predictor
        cls.mock_genre_predictor = MagicMock(spec=GenrePredictor)
        cls.mock_genre_predictor.predict.return_value = [['Sci-Fi', 'Adventure']]
    
    @classmethod
    def tearDownClass(cls):
        """Clean up test data."""
        shutil.rmtree(cls.test_dir)
    
    def test_init(self):
        """Test initialization of GenreVectorPredictor."""
        predictor = GenreVectorPredictor(
            vector_db_path='test_path',
            genre_predictor_path='test_path',
            similarity_threshold=0.8,
            top_k=5,
            weight_vector=0.6,
            weight_ml=0.4
        )
        
        self.assertEqual(predictor.vector_db_path, 'test_path')
        self.assertEqual(predictor.genre_predictor_path, 'test_path')
        self.assertEqual(predictor.similarity_threshold, 0.8)
        self.assertEqual(predictor.top_k, 5)
        self.assertEqual(predictor.weight_vector, 0.6)
        self.assertEqual(predictor.weight_ml, 0.4)
    
    @patch('genre_vector_predictor.MovieVectorDB')
    def test_load_vector_db(self, mock_vector_db_class):
        """Test loading vector database."""
        # Setup mock
        mock_instance = mock_vector_db_class.return_value
        mock_instance.load.return_value = True
        
        # Test loading
        predictor = GenreVectorPredictor()
        result = predictor.load_vector_db('test_path')
        
        self.assertTrue(result)
        mock_instance.load.assert_called_once_with('test_path')
    
    @patch('genre_vector_predictor.GenrePredictor')
    def test_load_genre_predictor(self, mock_genre_predictor_class):
        """Test loading genre predictor."""
        # Setup mock
        mock_genre_predictor_class.load.return_value = MagicMock()
        
        # Test loading
        predictor = GenreVectorPredictor()
        result = predictor.load_genre_predictor('test_path')
        
        self.assertTrue(result)
        mock_genre_predictor_class.load.assert_called_once_with(models_dir='test_path')
    
    def test_predict_genre_vector(self):
        """Test predicting genres using vector similarity."""
        # Setup predictor with mock vector database
        predictor = GenreVectorPredictor()
        predictor.vector_db = self.mock_vector_db
        
        # Test prediction
        genres = predictor.predict_genre_vector('test query')
        
        self.mock_vector_db.search.assert_called_once()
        self.assertIsInstance(genres, list)
        self.assertIn('Sci-Fi', genres)
        self.assertIn('Adventure', genres)
    
    def test_predict_genre_ml(self):
        """Test predicting genres using ML model."""
        # Setup predictor with mock genre predictor
        predictor = GenreVectorPredictor()
        predictor.genre_predictor = self.mock_genre_predictor
        
        # Test prediction
        genres = predictor.predict_genre_ml({'plot': 'test plot'})
        
        self.mock_genre_predictor.predict.assert_called_once()
        self.assertIsInstance(genres, list)
        self.assertEqual(genres, ['Sci-Fi', 'Adventure'])
    
    def test_predict_genre_hybrid(self):
        """Test predicting genres using hybrid approach."""
        # Setup predictor with mock components
        predictor = GenreVectorPredictor(weight_vector=0.5, weight_ml=0.5)
        predictor.vector_db = self.mock_vector_db
        predictor.genre_predictor = self.mock_genre_predictor
        
        # Test prediction
        genres = predictor.predict_genre_hybrid('test query', {'plot': 'test plot'})
        
        self.mock_vector_db.search.assert_called_once()
        self.mock_genre_predictor.predict.assert_called_once()
        self.assertIsInstance(genres, list)
        self.assertIn('Sci-Fi', genres)
        self.assertIn('Adventure', genres)
    
    def test_save_and_load_config(self):
        """Test saving and loading configuration."""
        # Create a predictor with custom settings
        predictor = GenreVectorPredictor(
            similarity_threshold=0.85,
            top_k=15,
            weight_vector=0.7,
            weight_ml=0.3
        )
        
        # Save configuration to temporary directory
        config_dir = os.path.join(self.test_dir, 'config')
        save_info = predictor.save(config_dir)
        
        # Check that the file was created
        self.assertTrue(os.path.exists(os.path.join(config_dir, 'config.json')))
        
        # Load the configuration
        loaded_predictor = GenreVectorPredictor.load_config(config_dir)
        
        # Check that settings were preserved
        self.assertEqual(loaded_predictor.similarity_threshold, 0.85)
        self.assertEqual(loaded_predictor.top_k, 15)
        self.assertEqual(loaded_predictor.weight_vector, 0.7)
        self.assertEqual(loaded_predictor.weight_ml, 0.3)
    
    @patch('genre_vector_predictor.GenreVectorPredictor.predict_genre_vector')
    @patch('genre_vector_predictor.GenreVectorPredictor.predict_genre_ml')
    def test_evaluate(self, mock_predict_ml, mock_predict_vector):
        """Test evaluation functionality."""
        # Setup mocks
        mock_predict_vector.return_value = ['Sci-Fi', 'Adventure']
        mock_predict_ml.return_value = ['Sci-Fi', 'Action']
        
        # Create predictor
        predictor = GenreVectorPredictor()
        predictor.vector_db = self.mock_vector_db
        predictor.genre_predictor = self.mock_genre_predictor
        
        # Create test data for evaluation
        eval_data = pd.DataFrame({
            'plot': ['Test plot 1', 'Test plot 2'],
            'plot_lemmatized': ['test plot 1', 'test plot 2'],
            'genre_list': [['Sci-Fi', 'Adventure'], ['Action', 'Thriller']],
            'Origin/Ethnicity': ['American', 'British']
        })
        
        # Test vector-based evaluation
        metrics = predictor.evaluate(eval_data, method='vector')
        self.assertIn('f1_micro', metrics)
        self.assertIn('hamming_loss', metrics)
        mock_predict_vector.assert_called()
        
        # Test ML-based evaluation
        metrics = predictor.evaluate(eval_data, method='ml')
        self.assertIn('f1_micro', metrics)
        self.assertIn('hamming_loss', metrics)
        mock_predict_ml.assert_called()


if __name__ == '__main__':
    unittest.main()
