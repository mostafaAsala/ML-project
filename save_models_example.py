"""
Example script demonstrating how to save and load models using the ModelSaver class.

This script shows:
1. How to save all models from a ModelEvaluator
2. How to save only the best model
3. How to load a saved model and use it for predictions
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
    print("Loading and preparing data...")
    
    # Load the preprocessed data (adjust path as needed)
    df = pd.read_csv('wiki_movie_plots_deduped_cleaned.csv')
    print(f"Loaded dataset with {len(df)} movies")
    
    # Remove empty genre lists
    df = df[df['genre_list']!="[]"]
    
    # Create TF-IDF features from plot descriptions
    tfidf = TfidfVectorizer(
        max_features=5000,         # limit features for performance
        stop_words='english',      # remove common English stopwords
        ngram_range=(1, 2),        # unigrams and bigrams
        min_df=5,                  # ignore terms that appear in <5 documents
        max_df=0.8,                # ignore very frequent terms
        strip_accents='unicode',   # normalize accents
        sublinear_tf=True          # apply sublinear tf scaling
    )
    
    # Transform the plot_lemmatized column
    X_plot = tfidf.fit_transform(df['plot_lemmatized'])
    print(f"TF-IDF features shape: {X_plot.shape}")
    
    # Prepare the target variable (genre)
    mlb_genre = MultiLabelBinarizer()
    y = mlb_genre.fit_transform(df['genre_list'])
    print(f"Target shape: {y.shape}")
    
    # Add location features
    X_location = pd.get_dummies(df['Origin/Ethnicity'], prefix='loc')
    print(f"Location features shape: {X_location.shape}")
    
    # Combine all features
    X_combined = hstack([X_plot, csr_matrix(X_location.values)])
    print(f"Combined features shape: {X_combined.shape}")
    
    # Select top features based on chi-squared scores
    k = 2000  # Number of top features to select
    selector = SelectKBest(chi2, k=k)
    X_selected = selector.fit_transform(X_combined, y)
    print(f"Selected features shape: {X_selected.shape}")
    
    # Get feature names
    tfidf_feature_names = tfidf.get_feature_names_out()
    location_feature_names = X_location.columns.tolist()
    all_feature_names = list(tfidf_feature_names) + location_feature_names
    
    # Get selected feature names
    selected_indices = selector.get_support(indices=True)
    selected_feature_names = [all_feature_names[i] for i in selected_indices]
    
    print("Training and evaluating models...")
    
    # Create an instance of ModelEvaluator
    evaluator = ModelEvaluator(random_state=42)
    
    # Add models (using a subset for faster execution)
    evaluator.add_basic_models()
    evaluator.add_intermediate_models()
    
    # Evaluate models
    evaluator.evaluate_models(
        X=X_selected, 
        y=y, 
        test_size=0.2, 
        feature_names=selected_feature_names,
        target_names=mlb_genre.classes_
    )
    
    # Print summary
    evaluator.print_summary()
    
    print("\nSaving models to disk...")
    
    # Create a ModelSaver instance
    saver = ModelSaver(base_dir='saved_models')
    
    # Save all models
    save_info = saver.save_all_models(
        evaluator=evaluator,
        dataset_name='movie_genre_prediction',
        include_data=False
    )
    
    # Save only the best model
    best_model_info = saver.save_best_model(
        evaluator=evaluator,
        dataset_name='movie_genre_prediction'
    )
    
    print("\nDemonstrating how to load and use a saved model...")
    
    # Load the best model
    best_model_path = best_model_info['model_path']
    loaded_model = saver.load_model(best_model_path)
    
    # Split data for demonstration
    X_train, X_test, y_train, y_test = train_test_split(
        X_selected, y, test_size=0.2, random_state=42
    )
    
    # Make predictions with the loaded model
    predictions = loaded_model.predict(X_test)
    
    # Calculate accuracy
    accuracy = np.mean(np.all(predictions == y_test, axis=1))
    print(f"Loaded model accuracy: {accuracy:.4f}")
    
    print("\nModel saving and loading demonstration complete!")

if __name__ == "__main__":
    main()
