"""
Movie Vector Embedding Visualization

This module provides visualization tools for movie vector embeddings to understand genre clustering
and explore the vector space created by the MovieVectorDB.

Features:
1. Dimensionality reduction for high-dimensional embeddings (PCA, t-SNE, UMAP)
2. 2D and 3D visualizations of movie embeddings
3. Genre clustering analysis
4. Interactive plots for exploration

Usage:
    from movie_vector_visualizer import MovieVectorVisualizer
    visualizer = MovieVectorVisualizer(movie_vector_db)
    visualizer.visualize_2d()  # Create a 2D visualization
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Optional, Tuple, Union, Any
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import plotly.express as px
import plotly.graph_objects as go
from movie_vector_db import MovieVectorDB

class MovieVectorVisualizer:
    """
    A class for visualizing movie vector embeddings and analyzing genre clustering.
    """

    def __init__(self, vector_db: MovieVectorDB):
        """
        Initialize the visualizer with a MovieVectorDB instance.

        Args:
            vector_db: A MovieVectorDB instance with loaded data and embeddings
        """
        self.vector_db = vector_db
        self.embeddings = None
        self.movies_df = None
        self.reduced_embeddings = {}
        self.output_dir = "visualizations"

        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)

        # Load data from vector_db
        self._load_data_from_db()

    def _load_data_from_db(self) -> None:
        """
        Load embeddings and movie data from the vector database.
        """
        if self.vector_db.index is None:
            raise ValueError("Vector database index not built. Call build_index() first.")

        if self.vector_db.embeddings is None:
            raise ValueError("No embeddings found in vector database.")

        self.embeddings = self.vector_db.embeddings
        self.movies_df = self.vector_db.movies_df

        # Ensure genre column exists and is processed
        if 'genre' in self.movies_df.columns:
            # Split multi-genre movies into primary genre
            self.movies_df['primary_genre'] = self.movies_df['genre'].apply(
                lambda x: x.split(',')[0].strip() if isinstance(x, str) and ',' in x else x
            )
        else:
            print("Warning: No genre column found in the movie data.")

    def reduce_dimensions(self, method: str = 'pca', n_components: int = 2,
                         perplexity: int = 30, random_state: int = 42) -> np.ndarray:
        """
        Reduce the dimensionality of embeddings for visualization.

        Args:
            method: Dimensionality reduction method ('pca', 'tsne')
            n_components: Number of dimensions to reduce to (2 or 3)
            perplexity: Perplexity parameter for t-SNE
            random_state: Random state for reproducibility

        Returns:
            Reduced embeddings as a NumPy array
        """
        if self.embeddings is None:
            self._load_data_from_db()

        # Check if we've already computed this reduction
        cache_key = f"{method}_{n_components}"
        if cache_key in self.reduced_embeddings:
            return self.reduced_embeddings[cache_key]

        print(f"Reducing dimensions using {method.upper()} to {n_components}D...")

        if method.lower() == 'pca':
            reducer = PCA(n_components=n_components, random_state=random_state)
            reduced = reducer.fit_transform(self.embeddings)
            explained_var = sum(reducer.explained_variance_ratio_) * 100
            print(f"Explained variance: {explained_var:.2f}%")

        elif method.lower() == 'tsne':
            reducer = TSNE(n_components=n_components, perplexity=perplexity,
                          random_state=random_state, n_iter=1000)
            reduced = reducer.fit_transform(self.embeddings)

        else:
            raise ValueError(f"Unsupported dimensionality reduction method: {method}")

        # Cache the result
        self.reduced_embeddings[cache_key] = reduced

        return reduced

    def visualize_2d(self, method: str = 'tsne', save_path: Optional[str] = None,
                    title: str = "Movie Embeddings Visualization",
                    interactive: bool = False) -> None:
        """
        Create a 2D visualization of movie embeddings colored by genre.

        Args:
            method: Dimensionality reduction method ('pca', 'tsne')
            save_path: Path to save the visualization (if None, will use default)
            title: Title for the visualization
            interactive: Whether to create an interactive plot with Plotly
        """
        # Reduce dimensions to 2D
        embeddings_2d = self.reduce_dimensions(method=method, n_components=2)

        # Prepare data for visualization
        plot_df = pd.DataFrame({
            'x': embeddings_2d[:, 0],
            'y': embeddings_2d[:, 1],
            'title': self.movies_df['title'],
            'year': self.movies_df['year'] if 'year' in self.movies_df.columns else None,
        })

        # Add genre information if available
        if 'primary_genre' in self.movies_df.columns:
            plot_df['genre'] = self.movies_df['primary_genre']
        elif 'genre' in self.movies_df.columns:
            plot_df['genre'] = self.movies_df['genre']

        # Create visualization
        if interactive:
            self._create_interactive_2d_plot(plot_df, method, title, save_path)
        else:
            self._create_static_2d_plot(plot_df, method, title, save_path)

    def _create_static_2d_plot(self, plot_df: pd.DataFrame, method: str,
                              title: str, save_path: Optional[str]) -> None:
        """Create a static 2D plot using matplotlib/seaborn."""
        plt.figure(figsize=(12, 10))

        if 'genre' in plot_df.columns:
            # Color by genre
            sns.scatterplot(data=plot_df, x='x', y='y', hue='genre', alpha=0.7, s=50)
            plt.legend(title='Genre', bbox_to_anchor=(1.05, 1), loc='upper left')
        else:
            # No genre information
            sns.scatterplot(data=plot_df, x='x', y='y', alpha=0.7, s=50)

        plt.title(f"{title} ({method.upper()})")
        plt.xlabel(f"{method.upper()} Component 1")
        plt.ylabel(f"{method.upper()} Component 2")
        plt.tight_layout()

        # Save the plot if a path is provided
        if save_path is None:
            save_path = os.path.join(self.output_dir, f"movie_embeddings_2d_{method}.png")

        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"2D visualization saved to {save_path}")

    def _create_interactive_2d_plot(self, plot_df: pd.DataFrame, method: str,
                                   title: str, save_path: Optional[str]) -> None:
        """Create an interactive 2D plot using Plotly."""
        hover_data = ['title']
        if 'year' in plot_df.columns:
            hover_data.append('year')
        plot_df = plot_df.explode('genre')
        if 'genre' in plot_df.columns:
            # Color by genre
            fig = px.scatter(
                plot_df, x='x', y='y', color='genre',
                hover_name='title', hover_data=hover_data,
                title=f"{title} ({method.upper()})"
            )
        else:
            # No genre information
            fig = px.scatter(
                plot_df, x='x', y='y',
                hover_name='title', hover_data=hover_data,
                title=f"{title} ({method.upper()})"
            )

        fig.update_layout(
            xaxis_title=f"{method.upper()} Component 1",
            yaxis_title=f"{method.upper()} Component 2",
            legend_title="Genre"
        )

        # Save the plot if a path is provided
        if save_path is None:
            save_path = os.path.join(self.output_dir, f"movie_embeddings_2d_{method}.html")

        fig.write_html(save_path)
        print(f"Interactive 2D visualization saved to {save_path}")

        # Show the plot
        fig.show()

    def visualize_3d(self, method: str = 'pca', save_path: Optional[str] = None,
                    title: str = "3D Movie Embeddings Visualization") -> None:
        """
        Create a 3D visualization of movie embeddings colored by genre.

        Args:
            method: Dimensionality reduction method ('pca', 'tsne')
            save_path: Path to save the visualization (if None, will use default)
            title: Title for the visualization
        """
        # Reduce dimensions to 3D
        embeddings_3d = self.reduce_dimensions(method=method, n_components=3)

        # Prepare data for visualization
        plot_df = pd.DataFrame({
            'x': embeddings_3d[:, 0],
            'y': embeddings_3d[:, 1],
            'z': embeddings_3d[:, 2],
            'title': self.movies_df['title'],
            'year': self.movies_df['year'] if 'year' in self.movies_df.columns else None,
        })

        # Add genre information if available
        if 'primary_genre' in self.movies_df.columns:
            plot_df['genre'] = self.movies_df['primary_genre']
        elif 'genre' in self.movies_df.columns:
            plot_df['genre'] = self.movies_df['genre']
        plot_df = plot_df.explode('genre')
        # Create interactive 3D plot with Plotly
        hover_data = ['title']
        if 'year' in plot_df.columns:
            hover_data.append('year')

        if 'genre' in plot_df.columns:
            # Color by genre
            fig = px.scatter_3d(
                plot_df, x='x', y='y', z='z', color='genre',
                hover_name='title', hover_data=hover_data,
                title=f"{title} ({method.upper()})"
            )
        else:
            # No genre information
            fig = px.scatter_3d(
                plot_df, x='x', y='y', z='z',
                hover_name='title', hover_data=hover_data,
                title=f"{title} ({method.upper()})"
            )

        fig.update_layout(
            scene=dict(
                xaxis_title=f"{method.upper()} Component 1",
                yaxis_title=f"{method.upper()} Component 2",
                zaxis_title=f"{method.upper()} Component 3",
            ),
            legend_title="Genre"
        )

        # Save the plot if a path is provided
        if save_path is None:
            save_path = os.path.join(self.output_dir, f"movie_embeddings_3d_{method}.html")

        fig.write_html(save_path)
        print(f"Interactive 3D visualization saved to {save_path}")

        # Show the plot
        fig.show()

    def visualize_genre_distribution(self, save_path: Optional[str] = None) -> None:
        """
        Visualize the distribution of genres in the dataset.

        Args:
            save_path: Path to save the visualization (if None, will use default)
        """
        if 'genre' not in self.movies_df.columns and 'primary_genre' not in self.movies_df.columns:
            print("No genre information available for visualization.")
            return

        # Use primary_genre if available, otherwise use genre
        genre_col = 'primary_genre' if 'primary_genre' in self.movies_df.columns else 'genre'
        df = self.movies_df.explode(genre_col)
        # Count genres
        genre_counts = df[genre_col].value_counts()

        # Filter to top 20 genres if there are too many
        if len(genre_counts) > 20:
            genre_counts = genre_counts.head(20)
            title = "Top 20 Genres Distribution"
        else:
            title = "Genre Distribution"

        # Create visualization
        plt.figure(figsize=(12, 8))
        sns.barplot(x=genre_counts.values, y=genre_counts.index)
        plt.title(title)
        plt.xlabel("Number of Movies")
        plt.ylabel("Genre")
        plt.tight_layout()

        # Save the plot if a path is provided
        if save_path is None:
            save_path = os.path.join(self.output_dir, "genre_distribution.png")

        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Genre distribution visualization saved to {save_path}")

    def visualize_genre_similarity(self, min_movies_per_genre: int = 10,
                                  save_path: Optional[str] = None) -> None:
        """
        Visualize the similarity between different genres based on their embeddings.

        Args:
            min_movies_per_genre: Minimum number of movies required for a genre to be included
            save_path: Path to save the visualization (if None, will use default)
        """
        if 'genre' not in self.movies_df.columns and 'primary_genre' not in self.movies_df.columns:
            print("No genre information available for visualization.")
            return

        # Use primary_genre if available, otherwise use genre
        genre_col = 'primary_genre' if 'primary_genre' in self.movies_df.columns else 'genre'

        # Filter genres with enough movies
        genre_counts = self.movies_df[genre_col].value_counts()
        valid_genres = genre_counts[genre_counts >= min_movies_per_genre].index.tolist()

        if len(valid_genres) < 2:
            print(f"Not enough genres with at least {min_movies_per_genre} movies.")
            return

        # Calculate average embedding for each genre
        genre_embeddings = {}
        for genre in valid_genres:
            genre_indices = self.movies_df[self.movies_df[genre_col] == genre].index.tolist()
            if genre_indices:
                genre_embedding = np.mean(self.embeddings[genre_indices], axis=0)
                genre_embeddings[genre] = genre_embedding

        # Calculate similarity matrix
        genres = list(genre_embeddings.keys())
        similarity_matrix = np.zeros((len(genres), len(genres)))

        for i, genre1 in enumerate(genres):
            for j, genre2 in enumerate(genres):
                # Cosine similarity
                dot_product = np.dot(genre_embeddings[genre1], genre_embeddings[genre2])
                norm1 = np.linalg.norm(genre_embeddings[genre1])
                norm2 = np.linalg.norm(genre_embeddings[genre2])
                similarity = dot_product / (norm1 * norm2)
                similarity_matrix[i, j] = similarity

        # Create heatmap
        plt.figure(figsize=(12, 10))
        sns.heatmap(similarity_matrix, annot=True, fmt=".2f", cmap="YlGnBu",
                   xticklabels=genres, yticklabels=genres)
        plt.title("Genre Similarity Based on Embeddings")
        plt.tight_layout()

        # Save the plot if a path is provided
        if save_path is None:
            save_path = os.path.join(self.output_dir, "genre_similarity.png")

        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Genre similarity visualization saved to {save_path}")

    def visualize_all(self, output_dir: Optional[str] = None) -> None:
        """
        Generate all visualizations and save them to the specified directory.

        Args:
            output_dir: Directory to save visualizations (if None, will use default)
        """
        if output_dir:
            self.output_dir = output_dir
            os.makedirs(self.output_dir, exist_ok=True)

        print("Generating all visualizations...")

        # 2D visualizations
        self.visualize_2d(method='pca', interactive=True)
        self.visualize_2d(method='tsne', interactive=True)

        # 3D visualization
        self.visualize_3d(method='pca')

        # Genre distribution
        self.visualize_genre_distribution()

        # Genre similarity
        self.visualize_genre_similarity()

        print(f"All visualizations saved to {self.output_dir}")


def main():
    """
    Main function to demonstrate the usage of MovieVectorVisualizer.
    """
    import argparse

    parser = argparse.ArgumentParser(description="Movie Vector Embedding Visualization")
    parser.add_argument("--data", type=str, default="wiki_movie_plots_cleaned.csv",
                        help="Path to movie data CSV file")
    parser.add_argument("--source", type=str, choices=["wiki", "tmdb"], default="wiki",
                        help="Source of the movie data")
    parser.add_argument("--create_db", action="store_true",
                        help="Create a new vector database if one doesn't exist")
    parser.add_argument("--output_dir", type=str, default="visualizations",
                        help="Directory to save visualizations")
    parser.add_argument("--method", type=str, choices=["pca", "tsne"], default="tsne",
                        help="Dimensionality reduction method")
    parser.add_argument("--interactive", action="store_true",
                        help="Create interactive visualizations with Plotly")
    parser.add_argument("--all", action="store_true",
                        help="Generate all visualizations")
    args = parser.parse_args()

    # Import here to avoid circular imports
    from movie_vector_db import MovieVectorDB

    # Initialize the vector database
    db = MovieVectorDB()

    # Try to load existing database
    success = db.load()

    if not success or args.create_db:
        print(f"Creating vector database from {args.data}...")
        db.load_data(args.data, data_source=args.source)
        db.preprocess_data()
        db.create_embeddings()
        db.build_index()
        db.save()
    elif not success:
        print("No existing database found. Use --create_db to create a new one.")
        return

    # Create visualizer
    visualizer = MovieVectorVisualizer(db)

    # Set output directory
    if args.output_dir:
        visualizer.output_dir = args.output_dir
        os.makedirs(visualizer.output_dir, exist_ok=True)

    # Generate visualizations
    if args.all:
        visualizer.visualize_all()
    else:
        # 2D visualization
        visualizer.visualize_2d(method=args.method, interactive=args.interactive)

        # Genre distribution
        visualizer.visualize_genre_distribution()

    print(f"Visualizations saved to {visualizer.output_dir}")


if __name__ == "__main__":
    main()
