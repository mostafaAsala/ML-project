"""
Example script demonstrating how to use the GenreHybridPredictor class.

This script shows:
1. How to initialize the GenreHybridPredictor
2. How to predict genres using the hybrid approach
3. How to evaluate prediction performance
4. How to save and load the predictor configuration
"""

import pandas as pd
import argparse
from genre_hybrid_predictor import GenreHybridPredictor
from genre_vector_predictor import GenreVectorPredictor
from genre_predictor import GenrePredictor


def predict_example():
    """
    Example of predicting genres using the GenreHybridPredictor.
    """
    print("=== Genre Hybrid Prediction Example ===")
    
    # Initialize the predictor
    predictor = GenreHybridPredictor()
    
    # Load the vector database and genre predictor
    predictor.vector_predictor.load_vector_db()
    predictor.load_ml_predictor()
    
    # Example movie descriptions
    example_movies = [
        {
            "title": "Space Adventure",
            "plot": "Astronauts embark on a dangerous mission to save humanity by traveling through a wormhole to find a new habitable planet.",
            "plot_lemmatized": "astronaut embark dangerous mission save humanity travel wormhole find new habitable planet",
            "Origin/Ethnicity": "American"
        },
        {
            "title": "Love in Paris",
            "plot": "Two strangers meet in Paris and fall in love over a weekend, but they must return to their separate lives on Monday.",
            "plot_lemmatized": "two stranger meet paris fall love weekend must return separate life monday",
            "Origin/Ethnicity": "French"
        },
        {
            "title": "The Heist",
            "plot": "A team of skilled thieves plan the biggest bank robbery in history, but things go wrong when one of them betrays the group.",
            "plot_lemmatized": "team skilled thief plan big bank robbery history thing go wrong one betray group",
            "Origin/Ethnicity": "American"
        }
    ]
    
    # Convert to DataFrame
    example_df = pd.DataFrame(example_movies)
    
    # Make predictions
    print("\n=== Hybrid Predictions ===")
    for i, (_, movie) in enumerate(example_df.iterrows()):
        print(f"\nMovie {i+1}: {movie['title']}")
        print(f"Plot: {movie['plot'][:100]}...")
        
        # Vector-based prediction
        vector_genres = predictor.vector_predictor.predict_genre_vector(movie['plot'])
        print(f"Vector-based genres: {vector_genres}")
        
        # ML-based prediction
        ml_genres = predictor.ml_predictor.predict(movie.to_dict())
        if isinstance(ml_genres, list) and len(ml_genres) > 0 and isinstance(ml_genres[0], list):
            ml_genres = ml_genres[0]
        print(f"ML-based genres: {ml_genres}")
        
        # Hybrid prediction
        hybrid_genres = predictor.predict_hybrid(movie['plot'], movie.to_dict())
        print(f"Hybrid genres: {hybrid_genres}")
    
    return predictor


def evaluate_example(data_path="wiki_movie_plots_deduped_cleaned.csv", sample_size=100):
    """
    Example of evaluating the GenreHybridPredictor.
    
    Args:
        data_path: Path to the movie data CSV file
        sample_size: Number of movies to use for evaluation
    """
    print(f"=== Evaluation Example (using {sample_size} movies) ===")
    
    # Initialize the predictor
    predictor = GenreHybridPredictor()
    
    # Load the vector database and genre predictor
    predictor.vector_predictor.load_vector_db()
    predictor.load_ml_predictor()
    
    # Load test data
    print(f"Loading test data from {data_path}...")
    df = pd.read_csv(data_path)
    
    # Take a sample for faster evaluation
    test_df = df.sample(n=sample_size, random_state=42)
    
    # Evaluate using different methods
    print("\nEvaluating vector-based prediction...")
    vector_metrics = predictor.evaluate(test_df, plot_col='plot_lemmatized', 
                                       genre_col='genre_list', method='vector')
    
    print("\nEvaluating ML-based prediction...")
    ml_metrics = predictor.evaluate(test_df, plot_col='plot_lemmatized', 
                                   genre_col='genre_list', method='ml')
    
    print("\nEvaluating hybrid prediction...")
    hybrid_metrics = predictor.evaluate(test_df, plot_col='plot_lemmatized', 
                                       genre_col='genre_list', method='hybrid')
    
    # Display results
    print("\n=== Evaluation Results ===")
    print("\nVector-based prediction:")
    for metric, value in vector_metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    print("\nML-based prediction:")
    for metric, value in ml_metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    print("\nHybrid prediction:")
    for metric, value in hybrid_metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    return predictor


def save_load_example():
    """
    Example of saving and loading the GenreHybridPredictor.
    """
    print("=== Save and Load Example ===")
    
    # Initialize the predictor
    predictor = GenreHybridPredictor(
        similarity_threshold=0.75,
        top_k=15,
        weight_vector=0.6,
        weight_ml=0.4
    )
    
    # Save the configuration
    save_info = predictor.save()
    print(f"Saved configuration: {save_info['config']}")
    
    # Load the configuration
    loaded_predictor = GenreHybridPredictor.load_config()
    print("\nLoaded configuration:")
    print(f"  similarity_threshold: {loaded_predictor.similarity_threshold}")
    print(f"  top_k: {loaded_predictor.top_k}")
    print(f"  weight_vector: {loaded_predictor.weight_vector}")
    print(f"  weight_ml: {loaded_predictor.weight_ml}")
    
    return loaded_predictor


def main():
    """Main function to run examples based on command line arguments."""
    parser = argparse.ArgumentParser(description="Genre Hybrid Predictor Example")
    parser.add_argument("--predict", action="store_true", help="Run prediction example")
    parser.add_argument("--evaluate", action="store_true", help="Run evaluation example")
    parser.add_argument("--data", default="wiki_movie_plots_deduped_cleaned.csv", help="Path to movie data CSV file")
    parser.add_argument("--sample", type=int, default=100, help="Number of movies to use for evaluation")
    parser.add_argument("--save-load", action="store_true", help="Run save and load example")
    
    args = parser.parse_args()
    
    # If no arguments provided, run all examples
    if not (args.predict or args.evaluate or args.save_load):
        args.predict = True
        args.evaluate = True
        args.save_load = True
    
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
