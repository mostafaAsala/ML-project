"""
Example script demonstrating how to use the GenreVectorPredictor class.

This script shows:
1. How to initialize the GenreVectorPredictor
2. How to predict genres using vector similarity
3. How to evaluate prediction performance
4. How to save and load the predictor configuration
"""

import pandas as pd
import argparse
from genre_vector_predictor import GenreVectorPredictor
from movie_vector_db import MovieVectorDB


def create_vector_database(data_path, data_source="wiki"):
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


def predict_example():
    """
    Example of predicting genres using the GenreVectorPredictor.
    """
    print("=== Genre Vector Prediction Example ===")

    # Initialize the predictor
    predictor = GenreVectorPredictor()

    # Load the vector database
    predictor.load_vector_db()

    # Example movie descriptions
    example_movies = [
        {
            "title": "Space Adventure",
            "plot": "Astronauts embark on a dangerous mission to save humanity by traveling through a wormhole to find a new habitable planet."
        },
        {
            "title": "Love in Paris",
            "plot": "Two strangers meet in Paris and fall in love over a weekend, but they must return to their separate lives on Monday."
        },
        {
            "title": "The Heist",
            "plot": "A team of skilled thieves plan the biggest bank robbery in history, but things go wrong when one of them betrays the group."
        }
    ]

    # Make predictions
    print("\n=== Vector-based Predictions ===")
    for i, movie in enumerate(example_movies):
        print(f"\nMovie {i+1}: {movie['title']}")
        print(f"Plot: {movie['plot'][:100]}...")

        # Vector-based prediction
        vector_genres = predictor.predict_genre_vector(movie['plot'])
        print(f"Predicted genres: {vector_genres}")

    return predictor


def evaluate_example(data_path="wiki_movie_plots_deduped_cleaned.csv", sample_size=100):
    """
    Example of evaluating the GenreVectorPredictor.

    Args:
        data_path: Path to the movie data CSV file
        sample_size: Number of movies to use for evaluation
    """
    print(f"=== Evaluation Example (using {sample_size} movies) ===")

    # Initialize the predictor
    predictor = GenreVectorPredictor()

    # Load the vector database
    predictor.load_vector_db()

    # Load test data
    print(f"Loading test data from {data_path}...")
    df = pd.read_csv(data_path)

    # Take a sample for faster evaluation
    test_df = df.sample(n=sample_size, random_state=42)

    # Evaluate
    print("\nEvaluating vector-based prediction...")
    vector_metrics = predictor.evaluate(test_df, plot_col='plot_lemmatized',
                                       genre_col='genre_list')

    # Display results
    print("\n=== Evaluation Results ===")
    print("\nVector-based prediction:")
    for metric, value in vector_metrics.items():
        print(f"  {metric}: {value:.4f}")

    return predictor


def save_load_example():
    """
    Example of saving and loading the GenreVectorPredictor.
    """
    print("=== Save and Load Example ===")

    # Initialize the predictor
    predictor = GenreVectorPredictor(
        similarity_threshold=0.75,
        top_k=15
    )

    # Save the configuration
    save_info = predictor.save()
    print(f"Saved configuration: {save_info['config']}")

    # Load the configuration
    loaded_predictor = GenreVectorPredictor.load_config()
    print("\nLoaded configuration:")
    print(f"  similarity_threshold: {loaded_predictor.similarity_threshold}")
    print(f"  top_k: {loaded_predictor.top_k}")

    return loaded_predictor


def main():
    """Main function to run examples based on command line arguments."""
    parser = argparse.ArgumentParser(description="Genre Vector Predictor Example")
    parser.add_argument("--create-db", action="store_true", help="Create a new vector database")
    parser.add_argument("--data", default="wiki_movie_plots_deduped_cleaned.csv", help="Path to movie data CSV file")
    parser.add_argument("--source", default="wiki", choices=["wiki", "tmdb"], help="Data source type")
    parser.add_argument("--predict", action="store_true", help="Run prediction example")
    parser.add_argument("--evaluate", action="store_true", help="Run evaluation example")
    parser.add_argument("--sample", type=int, default=100, help="Number of movies to use for evaluation")
    parser.add_argument("--save-load", action="store_true", help="Run save and load example")

    args = parser.parse_args()

    # If no arguments provided, run all examples
    if not (args.create_db or args.predict or args.evaluate or args.save_load):
        args.predict = True
        args.evaluate = True
        args.save_load = True

    # Create vector database if requested
    if args.create_db:
        create_vector_database(args.data, args.source)

    # Run prediction example
    if args.predict:
        predict_example()

    # Run evaluation example
    if args.evaluate:
        evaluate_example(args.data, args.sample)

    # Run save and load example
    if args.save_load:
        save_load_example()


if __name__ == "__main__":
    main()
