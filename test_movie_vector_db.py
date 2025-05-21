"""
Test script for the MovieVectorDB implementation.

This script tests the basic functionality of the MovieVectorDB class.
"""

import os
import unittest
import tempfile
import shutil
import numpy as np
import pandas as pd
from movie_vector_db import MovieVectorDB

class TestMovieVectorDB(unittest.TestCase):
    """Test cases for MovieVectorDB class."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test data."""
        # Create a temporary directory for test data
        cls.test_dir = tempfile.mkdtemp()
        
        # Create a small test dataset
        cls.test_data = pd.DataFrame({
            'title': ['The Matrix', 'Inception', 'Interstellar', 'The Godfather', 'Pulp Fiction'],
            'year': [1999, 2010, 2014, 1972, 1994],
            'director': ['Wachowski Brothers', 'Christopher Nolan', 'Christopher Nolan', 
                        'Francis Ford Coppola', 'Quentin Tarantino'],
            'genre': ['Sci-Fi', 'Sci-Fi', 'Sci-Fi', 'Crime', 'Crime'],
            'plot': [
                'A computer hacker learns about the true nature of reality.',
                'A thief who steals corporate secrets through dream-sharing technology.',
                'A team of explorers travel through a wormhole in space.',
                'The aging patriarch of an organized crime dynasty transfers control to his son.',
                'The lives of two mob hitmen, a boxer, and a pair of diner bandits intertwine.'
            ]
        })
        
        # Save test data to CSV
        cls.test_csv = os.path.join(cls.test_dir, 'test_movies.csv')
        cls.test_data.to_csv(cls.test_csv, index=False)
    
    @classmethod
    def tearDownClass(cls):
        """Clean up test data."""
        shutil.rmtree(cls.test_dir)
    
    def test_load_data(self):
        """Test loading data from CSV."""
        db = MovieVectorDB()
        df = db.load_data(self.test_csv, data_source="wiki")
        
        self.assertEqual(len(df), 5)
        self.assertIn('title', df.columns)
        self.assertIn('plot', df.columns)
    
    def test_preprocess_data(self):
        """Test data preprocessing."""
        db = MovieVectorDB()
        db.load_data(self.test_csv, data_source="wiki")
        df = db.preprocess_data()
        
        self.assertIn('text_for_embedding', df.columns)
        self.assertTrue(all(df['text_for_embedding'].str.contains('Title:')))
    
    def test_create_embeddings(self):
        """Test embedding creation."""
        db = MovieVectorDB()
        db.load_data(self.test_csv, data_source="wiki")
        db.preprocess_data()
        embeddings = db.create_embeddings()
        
        self.assertEqual(embeddings.shape[0], 5)  # 5 movies
        self.assertTrue(embeddings.shape[1] > 0)  # Embedding dimension
    
    def test_build_index(self):
        """Test index building."""
        db = MovieVectorDB()
        db.load_data(self.test_csv, data_source="wiki")
        db.preprocess_data()
        db.create_embeddings()
        index = db.build_index()
        
        self.assertEqual(index.ntotal, 5)  # 5 movies in the index
    
    def test_save_load(self):
        """Test saving and loading the database."""
        # Create and save a database
        db1 = MovieVectorDB()
        db1.load_data(self.test_csv, data_source="wiki")
        db1.preprocess_data()
        db1.create_embeddings()
        db1.build_index()
        
        save_path = os.path.join(self.test_dir, 'test_db')
        db1.save(save_path)
        
        # Load the database
        db2 = MovieVectorDB()
        success = db2.load(save_path)
        
        self.assertTrue(success)
        self.assertEqual(db2.index.ntotal, 5)
        self.assertEqual(len(db2.movies_df), 5)
    
    def test_search(self):
        """Test search functionality."""
        db = MovieVectorDB()
        db.load_data(self.test_csv, data_source="wiki")
        db.preprocess_data()
        db.create_embeddings()
        db.build_index()
        
        # Search for sci-fi movies
        results = db.search("science fiction movie", k=2)
        
        self.assertEqual(len(results), 2)
        # The top results should be sci-fi movies
        self.assertTrue(any(movie['genre'] == 'Sci-Fi' for movie in results))
        
        # Search for crime movies
        results = db.search("crime movie about gangsters", k=2)
        
        self.assertEqual(len(results), 2)
        # The top results should include The Godfather
        self.assertTrue(any(movie['title'] == 'The Godfather' for movie in results))

if __name__ == '__main__':
    unittest.main()
