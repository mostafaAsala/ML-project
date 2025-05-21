"""
Example script demonstrating how to load saved models using the ModelSaver class.

This script shows:
1. How to find all saved model directories
2. How to load models into a ModelEvaluator instance
3. How to use the loaded models for predictions
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from scipy.sparse import hstack, csr_matrix

# Import our custom classes
from model_evaluator import ModelEvaluator
from model_saver import ModelSaver

def main():
    print("Creating ModelSaver instance...")
    saver = ModelSaver(base_dir='saved_models')
    
    # Find all saved model directories
    print("\nFinding saved model directories...")
    saved_dirs = saver.find_saved_models()
    
    if not saved_dirs:
        print("No saved models found. Please run save_models_example.py first.")
        return
    
    print(f"Found {len(saved_dirs)} saved model directories:")
    for i, info in enumerate(saved_dirs):
        model_type = "Best model only" if info['is_best_only'] else f"{info['model_count']} models"
        print(f"{i+1}. {info['name']} ({model_type}) - {info['timestamp']}")
    
    # Choose the first directory (most recent)
    chosen_dir = saved_dirs[0]['directory']
    print(f"\nLoading models from: {chosen_dir}")
    
    # Load models into a new ModelEvaluator instance
    evaluator = saver.load_models_into_evaluator(chosen_dir)
    
    if evaluator is None:
        print("Failed to load models.")
        return
    
    # Print summary of loaded models
    print("\nSummary of loaded models:")
    summary = evaluator.print_summary()
    
    # Demonstrate using a loaded model for predictions
    print("\nDemonstrating how to use loaded models for predictions...")
    
    # Load a small sample of data for demonstration
    print("Loading and preparing a small sample of data...")
    
    try:
        # Load the preprocessed data (adjust path as needed)
        df = pd.read_csv('wiki_movie_plots_deduped_cleaned.csv')
        print(f"Loaded dataset with {len(df)} movies")
        
        # Take a small sample for demonstration
        df = df.sample(n=min(100, len(df)), random_state=42)
        
        # Remove empty genre lists
        df = df[df['genre_list']!="[]"]
        
        # Create TF-IDF features from plot descriptions
        tfidf = TfidfVectorizer(
            max_features=5000,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=5,
            max_df=0.8,
            strip_accents='unicode',
            sublinear_tf=True
        )
        
        # Transform the plot_lemmatized column
        X_plot = tfidf.fit_transform(df['plot_lemmatized'])
        
        # Prepare the target variable (genre)
        mlb_genre = MultiLabelBinarizer()
        y = mlb_genre.fit_transform(df['genre_list'])
        
        # Add location features
        X_location = pd.get_dummies(df['Origin/Ethnicity'], prefix='loc')
        
        # Combine all features
        X_combined = hstack([X_plot, csr_matrix(X_location.values)])
        
        # Select top features based on chi-squared scores
        k = 2000  # Number of top features to select
        selector = SelectKBest(chi2, k=k)
        X_selected = selector.fit_transform(X_combined, y)
        
        # Make predictions with the best model
        best_model = evaluator.best_model
        if best_model is not None:
            print(f"\nMaking predictions with the best model: {evaluator.best_model_name}")
            try:
                predictions = best_model.predict(X_selected)
                print(f"Predictions shape: {predictions.shape}")
                
                # Show a few example predictions
                print("\nExample predictions (first 3 samples):")
                for i in range(min(3, len(predictions))):
                    predicted_genres = [mlb_genre.classes_[j] for j in range(len(mlb_genre.classes_)) if predictions[i, j]]
                    actual_genres = [mlb_genre.classes_[j] for j in range(len(mlb_genre.classes_)) if y[i, j]]
                    print(f"Sample {i+1}:")
                    print(f"  Predicted genres: {predicted_genres}")
                    print(f"  Actual genres: {actual_genres}")
            except Exception as e:
                print(f"Error making predictions: {e}")
                print("This is expected if the feature dimensions don't match the model's expectations.")
                print("For accurate predictions, you need to use the same feature extraction pipeline as during training.")
        else:
            print("No best model found in the loaded evaluator.")
    
    except Exception as e:
        print(f"Error loading or processing data: {e}")
        print("This is just a demonstration - to make actual predictions, you need to use the same data preprocessing pipeline as during training.")
    
    print("\nModel loading demonstration complete!")

if __name__ == "__main__":
    main()
