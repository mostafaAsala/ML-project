"""
Movie Vector Database and Search Implementation

This module provides a vector database implementation for movie data, allowing for:
1. Loading and preprocessing movie data
2. Creating vector embeddings for movies
3. Building a searchable vector database
4. Performing semantic searches based on user queries

The implementation uses sentence-transformers for embeddings and FAISS for efficient similarity search.
"""

import os
import json
import pickle
import re
import numpy as np
import pandas as pd
from typing import List, Dict, Union, Tuple, Optional, Any
from tqdm import tqdm
import faiss
from sentence_transformers import SentenceTransformer

class MovieVectorDB:
    """
    A vector database for movies that enables semantic search based on movie descriptions,
    titles, genres, and other metadata.
    """
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize the MovieVectorDB with a specific embedding model.
        
        Args:
            model_name: The name of the sentence-transformer model to use for embeddings
        """
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.movies_df = None
        self.movie_ids = None
        self.embeddings = None
        self.db_path = "saved_models/vector_db"
        
        # Create directory if it doesn't exist
        os.makedirs(self.db_path, exist_ok=True)
    
    def load_data(self, file_path: str, data_source: str = "wiki") -> pd.DataFrame:
        """
        Load movie data from a CSV file.
        
        Args:
            file_path: Path to the CSV file containing movie data
            data_source: Source of the data ('wiki' or 'tmdb')
            
        Returns:
            DataFrame containing the loaded movie data
        """
        print(f"Loading movie data from {file_path}...")
        
        if data_source == "wiki":
            # Load wiki_movie_plots data
            df = pd.read_csv(file_path)
            # Rename columns to standardized format
            df = df.rename(columns={
                'Release Year': 'year',
                'Title': 'title',
                'Origin/Ethnicity': 'origin',
                'Director': 'director',
                'Cast': 'cast',
                'Genre': 'genre',
                'Wiki Page': 'wiki_url',
                'Plot': 'plot'
            })
            # Convert year to numeric
            df['year'] = pd.to_numeric(df['year'], errors='coerce')
            # Convert genre to list

                        # Define all known genres and their aliases
            known_genres = {
                'action': ['action'],
                'adventure': ['adventure'],
                'animation': ['animation', 'animated'],
                'biography': ['biography', 'bio'],
                'comedy': ['comedy', 'comedies'],
                'crime': ['crime'],
                'documentary': ['documentary', 'doc'],
                'drama': ['drama'],
                'family': ['family'],
                'fantasy': ['fantasy'],
                'history': ['history', 'historical'],
                'horror': ['horror'],
                'music': ['music', 'musical', 'musicals'],
                'mystery': ['mystery'],
                'romance': ['romance', 'romantic'],
                'sci-fi': ['sci-fi', 'science fiction', 'scifi', 'science-fiction'],
                'sport': ['sport', 'sports'],
                'thriller': ['thriller'],
                'war': ['war'],
                'western': ['western'],
                'unknown': ['unknown']
            }

            # Create a mapping from alias to canonical genre
            alias_to_genre = {}
            for genre, aliases in known_genres.items():
                for alias in aliases:
                    alias_to_genre[alias.lower()] = genre

            def extract_genres_from_string(genre_str):
                if not isinstance(genre_str, str):
                    return ['unknown']
                # Lowercase and split by comma or slash or semicolon
                tokens = re.split(r'[,/;]', genre_str.lower())
                genres_found = set()
                for token in tokens:
                    token = token.strip()
                    if token in alias_to_genre:
                        genres_found.add(alias_to_genre[token])
                    else:
                        # Try partial match for multi-word genres
                        for alias, genre in alias_to_genre.items():
                            if alias in token:
                                genres_found.add(genre)
                if not genres_found:
                    return ['unknown']
                return list(genres_found)

            # Apply to the 'Genre' column to create a clean genre list
            df['genre'] = df['genre'].apply(extract_genres_from_string)


        elif data_source == "tmdb":
            # Load TMDB data
            df = pd.read_csv(file_path)
            # Process JSON strings in cast and crew columns
            df['cast'] = df['cast'].apply(lambda x: json.loads(x) if isinstance(x, str) else [])
            df['crew'] = df['crew'].apply(lambda x: json.loads(x) if isinstance(x, str) else [])
            # Extract director names
            df['director'] = df['crew'].apply(
                lambda crew: ', '.join([person['name'] for person in crew 
                                       if person.get('job') == 'Director'])
            )
            # Extract cast names (top 5)
            df['cast_names'] = df['cast'].apply(
                lambda cast: ', '.join([person['name'] for person in cast[:5]])
                if isinstance(cast, list) and len(cast) > 0 else ''
            )
            # Rename movie_id to id for consistency
            df = df.rename(columns={'movie_id': 'id'})
        
        else:
            raise ValueError(f"Unsupported data source: {data_source}")
        
        self.movies_df = df
        print(f"Loaded {len(df)} movies.")
        return df
    
    def preprocess_data(self) -> pd.DataFrame:
        """
        Preprocess the movie data for embedding.
        
        Returns:
            Preprocessed DataFrame
        """
        if self.movies_df is None:
            raise ValueError("No data loaded. Call load_data() first.")
        
        print("Preprocessing movie data...")
        df = self.movies_df.copy()
        
        # Fill missing values
        for col in df.columns:
            if df[col].dtype == 'object':
                df[col] = df[col].fillna('')
        
        # Create text representation for embedding
        if 'plot' in df.columns:
            df['text_for_embedding'] = df.apply(
                lambda row: f"Title: {row['title']} " +
                           f"Director: {row['director']} " +
                           f"Genre: {row['genre']} " +
                           f"Year: {str(row['year'])} " +
                           f"Plot: {row['plot'][:1000]}",  # Limit plot length
                axis=1
            )
        else:
            # For TMDB data
            df['text_for_embedding'] = df.apply(
                lambda row: f"Title: {row['title']} " +
                           f"Director: {row['director']} " +
                           f"Cast: {row['cast_names']}",
                axis=1
            )
        
        self.movies_df = df
        return df
    
    def create_embeddings(self, batch_size: int = 32) -> np.ndarray:
        """
        Create embeddings for all movies in the dataset.
        
        Args:
            batch_size: Batch size for embedding creation
            
        Returns:
            NumPy array of embeddings
        """
        if self.movies_df is None:
            raise ValueError("No data loaded. Call load_data() first.")
        
        if 'text_for_embedding' not in self.movies_df.columns:
            self.preprocess_data()
        
        texts = self.movies_df['text_for_embedding'].tolist()
        print(f"Creating embeddings for {len(texts)} movies...")
        
        # Create embeddings in batches to avoid memory issues
        embeddings = []
        for i in tqdm(range(0, len(texts), batch_size)):
            batch_texts = texts[i:i+batch_size]
            batch_embeddings = self.model.encode(batch_texts)
            embeddings.append(batch_embeddings)
        
        # Combine all batches
        all_embeddings = np.vstack(embeddings)
        self.embeddings = all_embeddings
        
        # Store movie IDs for retrieval
        self.movie_ids = self.movies_df.index.tolist()
        
        print(f"Created embeddings with shape: {all_embeddings.shape}")
        return all_embeddings
    
    def build_index(self) -> faiss.Index:
        """
        Build a FAISS index for fast similarity search.
        
        Returns:
            FAISS index
        """
        if self.embeddings is None:
            self.create_embeddings()
        
        # Get embedding dimension
        dimension = self.embeddings.shape[1]
        
        # Create a flat index (exact search)
        index = faiss.IndexFlatL2(dimension)
        
        # Add vectors to the index
        index.add(self.embeddings.astype('float32'))
        
        self.index = index
        print(f"Built FAISS index with {index.ntotal} vectors")
        return index
    
    def save(self, path: Optional[str] = None) -> str:
        """
        Save the vector database to disk.
        
        Args:
            path: Directory path to save the database
            
        Returns:
            Path where the database was saved
        """
        if path is None:
            path = self.db_path
        
        os.makedirs(path, exist_ok=True)
        
        # Save the index
        if self.index is not None:
            faiss.write_index(self.index, os.path.join(path, "faiss_index.bin"))
        
        # Save the movie data
        if self.movies_df is not None:
            self.movies_df.to_pickle(os.path.join(path, "movies_df.pkl"))
        
        # Save movie IDs and other metadata
        metadata = {
            "model_name": self.model_name,
            "movie_ids": self.movie_ids,
        }
        
        with open(os.path.join(path, "metadata.pkl"), "wb") as f:
            pickle.dump(metadata, f)

        with open(os.path.join(path, "embeddings.pkl"), "wb") as f:
            pickle.dump(self.embeddings, f)
        
        print(f"Vector database saved to {path}")
        return path
    
    def load(self, path: Optional[str] = None) -> bool:
        """
        Load the vector database from disk.
        
        Args:
            path: Directory path to load the database from
            
        Returns:
            True if loaded successfully
        """
        if path is None:
            path = self.db_path
        
        try:
            # Load the index
            self.index = faiss.read_index(os.path.join(path, "faiss_index.bin"))
            
            # Load the movie data
            self.movies_df = pd.read_pickle(os.path.join(path, "movies_df.pkl"))
            
            # Load metadata
            with open(os.path.join(path, "metadata.pkl"), "rb") as f:
                metadata = pickle.load(f)
            with open(os.path.join(path, "embeddings.pkl"), "rb") as f:
                self.embeddings = pickle.load(f)
            
            self.model_name = metadata["model_name"]
            self.movie_ids = metadata["movie_ids"]
            
            # Load the model if needed
            if not hasattr(self, 'model') or self.model is None:
                self.model = SentenceTransformer(self.model_name)
            
            print(f"Vector database loaded from {path}")
            return True
        
        except Exception as e:
            print(f"Error loading vector database: {e}")
            return False
    
    def search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """
        Search for movies similar to the query.
        
        Args:
            query: Search query text
            k: Number of results to return
            
        Returns:
            List of dictionaries containing movie information
        """
        if self.index is None:
            raise ValueError("No index built. Call build_index() first.")
        
        # Encode the query
        query_embedding = self.model.encode([query])[0].reshape(1, -1).astype('float32')
        
        # Search the index
        distances, indices = self.index.search(query_embedding, k)
        
        # Get the movie information
        results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(self.movie_ids):
                movie_idx = self.movie_ids[idx]
                movie_data = self.movies_df.iloc[movie_idx].to_dict()
                movie_data['similarity_score'] = float(1 - distances[0][i] / 100)  # Normalize to 0-1
                results.append(movie_data)
        
        return results
