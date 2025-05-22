"""
Example script demonstrating how to use the GenrePredictor class.

This script shows:
1. How to load data and prepare features
2. How to train and evaluate models
3. How to save models and preprocessing components
4. How to load a saved predictor
5. How to make predictions on new data
"""

import pandas as pd
from genre_predictor import GenrePredictor

def train_and_save_example():
    """
    Example of training a new GenrePredictor and saving it.
    """
    print("=== Training a new GenrePredictor ===")
    
    # Create a new GenrePredictor
    predictor = GenrePredictor(random_state=42, models_dir='saved_models')
    
    # Load and prepare data
    predictor.load_data(
        file_path='wiki_movie_plots_deduped_cleaned.csv',
        plot_col='plot_lemmatized',
        genre_col='genre_list',
        location_col='Origin/Ethnicity'
    )
    
    # Prepare features
    predictor.prepare_features()
    
    # Train models (using only basic and intermediate for speed)
    predictor.train_models(model_levels=['basic', 'intermediate'])
    
    # Save models and preprocessing components
    save_info = predictor.save_models(
        dataset_name='movie_genre_predictor',
        save_all=True,
        include_data=False
    )
    
    print(f"Models and preprocessing components saved to {save_info['base_directory']}")
    
    return predictor

def load_and_predict_example():
    """
    Example of loading a saved GenrePredictor and making predictions.
    """
    print("\n=== Loading a saved GenrePredictor ===")
    
    # Load the saved predictor
    predictor = GenrePredictor.load(models_dir='saved_models')
    
    print(f"Loaded predictor with best model: {predictor.evaluator.best_model_name}")
    
    # Create some example movie data
    example_movies = [
        {
            'plot_lemmatized': "A superhero with extraordinary powers fights against an evil villain to save the world from destruction.",
            'Origin/Ethnicity': "American"
        },
        {
            'plot_lemmatized': "Two people meet and fall in love despite their different backgrounds and families who disapprove of their relationship.",
            'Origin/Ethnicity': "British"
        },
        {
            'plot_lemmatized': "A detective investigates a series of mysterious murders in a small town, uncovering dark secrets about the residents.",
            'Origin/Ethnicity': "French"
        },
        {
            'plot_lemmatized': "Astronauts travel to a distant planet where they discover an alien civilization and must find a way to communicate with them.",
            'Origin/Ethnicity': "American"
        }
    ]
    
    # Convert to DataFrame
    example_df = pd.DataFrame(example_movies)
    
    # Make predictions
    print("\nMaking predictions on example movies:")
    predictions = predictor.predict(example_df)
    
    # Display results
    for i, (movie, genres) in enumerate(zip(example_movies, predictions)):
        print(f"\nMovie {i+1}:")
        print(f"Plot: {movie['plot_lemmatized'][:100]}...")
        print(f"Origin: {movie['Origin/Ethnicity']}")
        print(f"Predicted genres: {genres}")
    
    return predictor

def end_to_end_example():
    """
    Complete end-to-end example of using GenrePredictor.
    """
    # Load a small sample of data for demonstration
    print("Loading a small sample of data for demonstration...")
    df = pd.read_csv('wiki_movie_plots_deduped_cleaned.csv')
    sample_df = df.sample(n=1000, random_state=42)
    sample_df.to_csv('sample_movie_data.csv', index=False)
    
    print("\n=== End-to-End Example ===")
    
    # Create a new GenrePredictor
    predictor = GenrePredictor(random_state=42, models_dir='saved_models')
    
    # Load and prepare data
    predictor.load_data(
        file_path='sample_movie_data.csv',
        plot_col='plot_lemmatized',
        genre_col='genre_list',
        location_col='Origin/Ethnicity'
    )
    
    # Prepare features
    predictor.prepare_features()
    
    # Train models (using only basic models for speed)
    predictor.train_models(model_levels=['basic'])
    
    # Save models and preprocessing components
    save_info = predictor.save_models(
        dataset_name='movie_genre_demo',
        save_all=True
    )
    
    print(f"Models saved to {save_info['base_directory']}")
    
    # Load the saved predictor
    loaded_predictor = GenrePredictor.load(models_dir='saved_models')
    
    # Select a few movies from the original dataset for testing
    test_movies = df.sample(n=5, random_state=100)[['plot_lemmatized', 'Origin/Ethnicity', 'genre_list']]
    
    # Make predictions
    print("\nMaking predictions on test movies:")
    predictions = loaded_predictor.predict(test_movies)
    
    # Display results
    for i, (_, movie) in enumerate(test_movies.iterrows()):
        print(f"\nMovie {i+1}:")
        print(f"Plot: {movie['plot_lemmatized'][:100]}...")
        print(f"Origin: {movie['Origin/Ethnicity']}")
        print(f"Actual genres: {movie['genre_list']}")
        print(f"Predicted genres: {predictions[i]}")
    
    return loaded_predictor

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "train":
            train_and_save_example()
        elif sys.argv[1] == "predict":
            load_and_predict_example()
        elif sys.argv[1] == "demo":
            end_to_end_example()
        else:
            print("Unknown command. Use 'train', 'predict', or 'demo'.")
    else:
        print("Running complete end-to-end example...")
        end_to_end_example()
