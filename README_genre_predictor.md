# Genre Predictor

The `GenrePredictor` class provides a complete pipeline for movie genre prediction, from data loading and preprocessing to model training, evaluation, saving, and loading.

## Features

- **Data Loading**: Load movie data from CSV files
- **Preprocessing**: Create TF-IDF features from plot descriptions and one-hot encoded location features
- **Feature Selection**: Select the most informative features using chi-squared test
- **Model Training**: Train and evaluate multiple machine learning models
- **Model Evaluation**: Compare model performance using various metrics
- **Model Saving**: Save trained models and preprocessing components to disk
- **Model Loading**: Load saved models and preprocessing components for later use
- **Prediction**: Make genre predictions on new movie data

## Installation

No additional installation is required beyond the dependencies already used in the project:
- Python 3.6+
- NumPy
- Pandas
- scikit-learn
- SciPy
- Pickle (part of Python's standard library)

## Usage

### Basic Usage

```python
from genre_predictor import GenrePredictor

# Create a new GenrePredictor
predictor = GenrePredictor()

# Load and prepare data
predictor.load_data('movie_data.csv')
predictor.prepare_features()

# Train models
predictor.train_models()

# Save models
predictor.save_models(dataset_name='movie_genre_predictor')

# Make predictions on new data
new_data = {...}  # Dictionary or DataFrame with plot_lemmatized and Origin/Ethnicity columns
predictions = predictor.predict(new_data)
```

### Loading a Saved Predictor

```python
from genre_predictor import GenrePredictor

# Load a saved predictor
predictor = GenrePredictor.load(models_dir='saved_models')

# Make predictions
predictions = predictor.predict(new_data)
```

## Example Scripts

### genre_predictor_example.py

This script demonstrates how to use the `GenrePredictor` class in different scenarios:

1. **Training and Saving**: `python genre_predictor_example.py train`
2. **Loading and Predicting**: `python genre_predictor_example.py predict`
3. **End-to-End Example**: `python genre_predictor_example.py demo`

## Detailed API

### GenrePredictor Class

#### Initialization

```python
GenrePredictor(random_state=42, n_jobs=-1, models_dir='saved_models')
```

- `random_state`: Random seed for reproducibility
- `n_jobs`: Number of CPU cores to use for parallel processing
- `models_dir`: Directory where models will be saved

#### Methods

##### load_data

```python
load_data(file_path, plot_col='plot_lemmatized', genre_col='genre_list', 
          location_col='Origin/Ethnicity', filter_empty_genres=True)
```

Loads and prepares the movie dataset.

- `file_path`: Path to the CSV file containing movie data
- `plot_col`: Column name containing the movie plot text
- `genre_col`: Column name containing the movie genres
- `location_col`: Column name containing the movie origin/ethnicity
- `filter_empty_genres`: Whether to filter out movies with empty genre lists

##### prepare_features

```python
prepare_features()
```

Prepares features from the loaded data:
1. Creates TF-IDF features from plot descriptions
2. Creates one-hot encoded location features
3. Combines all features
4. Selects the top features using chi-squared test

##### train_models

```python
train_models(model_levels='all', test_size=0.2)
```

Trains and evaluates models on the prepared features.

- `model_levels`: Which model levels to include: 'basic', 'intermediate', 'advanced', or 'all'
- `test_size`: Proportion of the dataset to include in the test split

##### save_models

```python
save_models(dataset_name=None, save_all=True, include_data=False)
```

Saves trained models to disk.

- `dataset_name`: Name of the dataset (used in directory naming)
- `save_all`: Whether to save all models or just the best model
- `include_data`: Whether to save the training data along with the models

##### predict

```python
predict(data, threshold=0.5)
```

Predicts genres for new movie data.

- `data`: DataFrame or dict containing plot and location columns
- `threshold`: Probability threshold for positive predictions

#### Class Methods

##### load

```python
GenrePredictor.load(models_dir, preprocessing_dir=None, models_save_dir=None)
```

Loads a GenrePredictor with trained models and preprocessing components.

- `models_dir`: Directory containing saved models
- `preprocessing_dir`: Directory containing preprocessing components. If None, will look for the most recent.
- `models_save_dir`: Directory to save new models. If None, uses the same as models_dir.

## Directory Structure

The `GenrePredictor` creates the following directory structure:

```
saved_models/
├── movie_genre_predictor_preprocessing_20230601_120000/
│   ├── tfidf_vectorizer.pkl
│   ├── multilabel_binarizer.pkl
│   ├── feature_selector.pkl
│   ├── column_names.json
│   ├── preprocessing_pipeline.pkl
│   └── preprocessing_info.json
└── movie_genre_predictor_20230601_120000/
    ├── model_summary.csv
    ├── feature_names.json
    ├── target_names.json
    ├── save_info.json
    ├── model1_name/
    │   ├── model1_name_model.pkl
    │   ├── model1_name_metrics.json
    │   └── model1_name_report.json
    ├── model2_name/
    │   ├── model2_name_model.pkl
    │   ├── model2_name_metrics.json
    │   └── model2_name_report.json
    └── ...
```

## Notes

- The `GenrePredictor` class integrates the functionality of `ModelEvaluator` and `ModelSaver` into a single pipeline.
- When making predictions on new data, make sure the data has the same structure as the training data.
- The preprocessing pipeline ensures that new data is transformed in the same way as the training data.
- Models are saved using Python's pickle module, which means they can only be loaded in a compatible Python environment.
