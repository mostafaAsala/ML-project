"""
Movie Vector Database and Search Example

This script demonstrates how to use the MovieVectorDB class to:
1. Load movie data
2. Create a vector database
3. Perform semantic searches based on user queries
4. Visualize movie embeddings to understand genre clustering

Usage:
    python movie_vector_search_example.py
    python movie_vector_search_example.py --visualize
"""

import os
import argparse
from typing import List, Dict, Any
from movie_vector_db import MovieVectorDB
from movie_vector_visualizer import MovieVectorVisualizer

def create_vector_database(data_path: str, data_source: str = "wiki") -> MovieVectorDB:
    """
    Create and save a vector database from movie data.

    Args:
        data_path: Path to the movie data CSV file
        data_source: Source of the data ('wiki' or 'tmdb')

    Returns:
        MovieVectorDB instance
    """
    print(f"Creating vector database from {data_path}...")

    # Initialize the vector database
    db = MovieVectorDB()

    # Load and preprocess the data
    db.load_data(data_path, data_source=data_source)
    db.preprocess_data()

    # Create embeddings and build the index
    db.create_embeddings()
    db.build_index()

    # Save the database
    db.save()

    print("Vector database created and saved successfully!")
    return db

def load_vector_database() -> MovieVectorDB:
    """
    Load a previously saved vector database.

    Returns:
        MovieVectorDB instance
    """
    print("Loading vector database...")

    # Initialize the vector database
    db = MovieVectorDB()

    # Load the database
    success = db.load()

    if success:
        print("Vector database loaded successfully!")
    else:
        print("Failed to load vector database.")

    return db

def search_movies(db: MovieVectorDB, query: str, k: int = 5) -> List[Dict[str, Any]]:
    """
    Search for movies using the vector database.

    Args:
        db: MovieVectorDB instance
        query: Search query
        k: Number of results to return

    Returns:
        List of movie results
    """
    print(f"Searching for: '{query}'")
    results = db.search(query, k=k)
    return results

def display_results(results: List[Dict[str, Any]]) -> None:
    """
    Display search results in a readable format.

    Args:
        results: List of movie results
    """
    if not results:
        print("No results found.")
        return

    print("\n===== SEARCH RESULTS =====")
    for i, movie in enumerate(results):
        print(f"\n--- Result #{i+1} (Score: {movie['similarity_score']:.2f}) ---")

        # Display title and year
        if 'year' in movie:
            print(f"Title: {movie['title']} ({movie['year']})")
        else:
            print(f"Title: {movie['title']}")

        # Display director
        if 'director' in movie and movie['director']:
            print(f"Director: {movie['director']}")

        # Display genre
        if 'genre' in movie and movie['genre']:
            print(f"Genre: {movie['genre']}")

        # Display cast
        if 'cast' in movie and movie['cast']:
            print(f"Cast: {movie['cast']}")
        elif 'cast_names' in movie and movie['cast_names']:
            print(f"Cast: {movie['cast_names']}")

        # Display plot summary (truncated)
        if 'plot' in movie and movie['plot']:
            plot = movie['plot']
            if len(plot) > 200:
                plot = plot[:200] + "..."
            print(f"Plot: {plot}")

def interactive_search(db: MovieVectorDB) -> None:
    """
    Run an interactive search loop.

    Args:
        db: MovieVectorDB instance
    """
    print("\n===== MOVIE VECTOR SEARCH =====")
    print("Type your search queries below. Enter 'quit' or 'exit' to end.")

    while True:
        query = input("\nEnter search query: ")

        if query.lower() in ['quit', 'exit', 'q']:
            print("Goodbye!")
            break

        if not query.strip():
            continue

        try:
            results = search_movies(db, query)
            display_results(results)
        except Exception as e:
            print(f"Error during search: {e}")

def visualize_embeddings(db: MovieVectorDB, method: str = 'tsne', interactive: bool = True) -> None:
    """
    Visualize movie embeddings using the MovieVectorVisualizer.

    Args:
        db: MovieVectorDB instance
        method: Dimensionality reduction method ('pca', 'tsne')
        interactive: Whether to create interactive visualizations
    """
    print("\n===== MOVIE EMBEDDING VISUALIZATION =====")

    # Create visualizer
    visualizer = MovieVectorVisualizer(db)

    # Generate visualizations
    print(f"Generating visualizations using {method.upper()}...")

    # 2D visualization
    visualizer.visualize_2d(method=method, interactive=interactive)

    # 3D visualization if interactive
    if interactive:
        visualizer.visualize_3d(method=method)

    # Genre distribution
    visualizer.visualize_genre_distribution()

    # Genre similarity
    visualizer.visualize_genre_similarity()

    print(f"Visualizations saved to {visualizer.output_dir}")

def main():
    """Main function to run the example."""
    parser = argparse.ArgumentParser(description="Movie Vector Database and Search Example")
    parser.add_argument("--create", action="store_true", help="Create a new vector database")
    parser.add_argument("--data", type=str, default="wiki_movie_plots_cleaned.csv",
                        help="Path to movie data CSV file")
    parser.add_argument("--source", type=str, choices=["wiki", "tmdb"], default="wiki",
                        help="Source of the movie data")
    parser.add_argument("--visualize", action="store_true",
                        help="Visualize movie embeddings")
    parser.add_argument("--method", type=str, choices=["pca", "tsne"], default="tsne",
                        help="Dimensionality reduction method for visualization")
    parser.add_argument("--interactive", action="store_true", default=True,
                        help="Create interactive visualizations")
    args = parser.parse_args()

    if args.create:
        db = create_vector_database(args.data, args.source)
    else:
        # Try to load existing database
        db = load_vector_database()

        # If loading fails, offer to create a new one
        if db.index is None:
            print("No existing database found.")
            create_new = input("Would you like to create a new vector database? (y/n): ")

            if create_new.lower() == 'y':
                db = create_vector_database(args.data, args.source)
            else:
                print("Exiting.")
                return

    # Visualize embeddings if requested
    if args.visualize:
        visualize_embeddings(db, method=args.method, interactive=args.interactive)
    else:
        # Run interactive search
        interactive_search(db)

if __name__ == "__main__":
    main()
