#please refer to genre_predictor_example.py for illustration of how to use it
import ast
import os
import pickle
import json
import numpy as np
import pandas as pd
from scipy.sparse import hstack, csr_matrix
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin

from model_evaluator import ModelEvaluator
from model_saver import ModelSaver


class LocationTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, location_categories):
        self.location_categories = location_categories

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_loc = pd.get_dummies(X, prefix='loc')

        # Ensure all expected categories are present
        for cat in self.location_categories:
            if cat not in X_loc.columns:
                X_loc[cat] = 0

        # Ensure only the expected categories are used, in the same order
        X_loc = X_loc[self.location_categories]

        # 🛠 Convert to float32 or float64 before passing to csr_matrix
        return csr_matrix(X_loc.values.astype('float32'))



class GenrePredictor:
    """
    A complete pipeline for movie genre prediction, including:
    - Data loading and preprocessing
    - Feature engineering (TF-IDF, location features)
    - Feature selection
    - Model training and evaluation
    - Model saving and loading

    This class integrates all the steps required to go from raw movie data to trained models
    and predictions.
    """

    def __init__(self, random_state=42, n_jobs=-1, models_dir='saved_models'):
        """
        Initialize the GenrePredictor.

        Parameters:
        -----------
        random_state : int, default=42
            Random seed for reproducibility
        n_jobs : int, default=-1
            Number of CPU cores to use for parallel processing
        models_dir : str, default='saved_models'
            Directory where models will be saved
        """
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.models_dir = models_dir

        # Initialize preprocessing components
        self.tfidf = TfidfVectorizer(
            max_features=300,
            stop_words='english',
            ngram_range=(1, 2),
            strip_accents='unicode',
            sublinear_tf=True
        )

        self.mlb = MultiLabelBinarizer()
        self.selector = SelectKBest(chi2, k=100)

        # Initialize model components
        self.evaluator = ModelEvaluator(random_state=random_state, n_jobs=n_jobs)
        self.saver = ModelSaver(base_dir=models_dir)

        # Placeholders for data and features
        self.df = None
        self.X_plot = None
        self.X_location = None
        self.X_combined = None
        self.X_selected = None
        self.y = None
        self.feature_names = None
        self.all_feature_names = None
        self.selected_feature_names = None

        # Create preprocessing pipeline
        self.pipeline = None

    def load_data(self, file_path, plot_col='plot_lemmatized', genre_col='genre_list',
                 location_col='Origin/Ethnicity', filter_empty_genres=True):
        """
        Load and prepare the movie dataset.

        Parameters:
        -----------
        file_path : str
            Path to the CSV file containing movie data
        plot_col : str, default='plot_lemmatized'
            Column name containing the movie plot text
        genre_col : str, default='genre_list'
            Column name containing the movie genres
        location_col : str, default='Origin/Ethnicity'
            Column name containing the movie origin/ethnicity
        filter_empty_genres : bool, default=True
            Whether to filter out movies with empty genre lists

        Returns:
        --------
        self : object
            Returns self
        """
        # Load the data
        self.df = pd.read_csv(file_path)
        print(f"Loaded dataset with {len(self.df)} movies")

        # Store column names
        self.plot_col = plot_col
        self.genre_col = genre_col
        self.location_col = location_col

        # Filter out movies with empty genre lists if requested
        if filter_empty_genres:
            self.df = self.df[self.df[genre_col]!="[]"]
            print(f"After filtering empty genres: {len(self.df)} movies")

        return self

    def prepare_features(self):
        """
        Prepare features from the loaded data.

        This method:
        1. Creates TF-IDF features from plot descriptions
        2. Creates one-hot encoded location features
        3. Combines all features
        4. Selects the top features using chi-squared test

        Returns:
        --------
        self : object
            Returns self
        """
        if self.df is None:
            raise ValueError("No data loaded. Call load_data() first.")

        # Create TF-IDF features from plot descriptions
        self.X_plot = self.tfidf.fit_transform(self.df[self.plot_col])
        print(f"TF-IDF features shape: {self.X_plot.shape}")

        # Prepare the target variable (genre)
        self.y = self.mlb.fit_transform(self.df[self.genre_col].apply(ast.literal_eval))
        print(f"Target shape: {self.y.shape}")
        print(f"Genre classes: {self.mlb.classes_}")

        # Add location features
        self.X_location = pd.get_dummies(self.df[self.location_col], prefix='loc')
        print(f"Location features shape: {self.X_location.shape}")
        
        # Combine all features
        self.X_combined = hstack([self.X_plot, csr_matrix(self.X_location.values)])
        print(f"Combined features shape: {self.X_combined.shape}")

        # Select top features
        self.X_selected = self.selector.fit_transform(self.X_combined, self.y)
        print(f"Selected features shape: {self.X_selected.shape}")

        # Get feature names
        tfidf_feature_names = self.tfidf.get_feature_names_out()
        location_feature_names = self.X_location.columns.tolist()
        self.all_feature_names = list(tfidf_feature_names) + location_feature_names
        
        # Get selected feature names
        selected_indices = self.selector.get_support(indices=True)
        self.selected_feature_names = [self.all_feature_names[i] for i in selected_indices]

        # Create the preprocessing pipeline
        self._create_pipeline()

        return self

    def _create_pipeline(self):
        """
        Create a scikit-learn pipeline for preprocessing new data.

        This pipeline will transform raw text and location data into the same
        feature space used for training.
        """
        
        # Use ColumnTransformer to apply transformers to specific columns
        preprocessor = ColumnTransformer([
            ('plot_tfidf', self.tfidf, self.plot_col),
            ('location', LocationTransformer(self.X_location.columns.tolist()), self.location_col)
        ])

        # Final pipeline
        self.pipeline = Pipeline([
            ('features', preprocessor),
            ('selector', self.selector)
        ])

        # Fit pipeline on the whole DataFrame (preprocessor handles column selection)
        self.pipeline.fit(self.df, self.y)

    def train_models(self, model_levels='all', test_size=0.2):
        """
        Train and evaluate models on the prepared features.

        Parameters:
        -----------
        model_levels : str or list, default='all'
            Which model levels to include: 'basic', 'intermediate', 'advanced', or 'all'
        test_size : float, default=0.2
            Proportion of the dataset to include in the test split

        Returns:
        --------
        self : object
            Returns self
        """
        if self.X_selected is None or self.y is None:
            raise ValueError("Features not prepared. Call prepare_features() first.")

        # Add models based on specified levels
        if (model_levels == 'all') or ('basic' in model_levels):
            print("Ading basic models...")
            self.evaluator.add_basic_models()

        if (model_levels == 'all') or ('intermediate' in model_levels):
            print("Adding intermediate models...")
            self.evaluator.add_intermediate_models()

        if (model_levels == 'all') or ('advanced' in model_levels):
            print("Adding advanced models...")
            self.evaluator.add_advanced_models()

        # Evaluate all models
        self.evaluator.evaluate_models(
            X=self.X_selected,
            y=self.y,
            test_size=test_size,
            feature_names=self.selected_feature_names,
            target_names=self.mlb.classes_
        )

        # Print summary
        self.summary = self.evaluator.print_summary()

        
        return self

    def save_models(self, dataset_name=None, save_all=True, include_data=False):
        """
        Save trained models to disk.

        Parameters:
        -----------
        dataset_name : str, default=None
            Name of the dataset (used in directory naming)
        save_all : bool, default=True
            Whether to save all models or just the best model
        include_data : bool, default=False
            Whether to save the training data along with the models

        Returns:
        --------
        dict : Information about saved models
        """
        if not self.evaluator.results:
            raise ValueError("No models trained. Call train_models() first.")

        # Save preprocessing components
        self._save_preprocessing_components(dataset_name)

        # Save models
        if save_all:
            save_info = self.saver.save_all_models(
                evaluator=self.evaluator,
                dataset_name=dataset_name,
                include_data=include_data,
                X=self.X_selected if include_data else None,
                y=self.y if include_data else None
            )
        else:
            save_info = self.saver.save_best_model(
                evaluator=self.evaluator,
                dataset_name=dataset_name
            )

        return save_info

    def _save_preprocessing_components(self, dataset_name=None):
        """
        Save preprocessing components (TF-IDF, MultiLabelBinarizer, SelectKBest).

        Parameters:
        -----------
        dataset_name : str, default=None
            Name of the dataset (used in directory naming)
        """
        # Create timestamp and directory name
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        if dataset_name:
            preprocessing_dir = os.path.join(self.models_dir, f"{dataset_name}_preprocessing_{timestamp}")
        else:
            preprocessing_dir = os.path.join(self.models_dir, f"preprocessing_{timestamp}")

        # Create the directory
        if not os.path.exists(preprocessing_dir):
            os.makedirs(preprocessing_dir)

        # Save TF-IDF vectorizer
        tfidf_path = os.path.join(preprocessing_dir, "tfidf_vectorizer.pkl")
        with open(tfidf_path, 'wb') as f:
            pickle.dump(self.tfidf, f)

        # Save MultiLabelBinarizer
        mlb_path = os.path.join(preprocessing_dir, "multilabel_binarizer.pkl")
        with open(mlb_path, 'wb') as f:
            pickle.dump(self.mlb, f)

        # Save SelectKBest
        selector_path = os.path.join(preprocessing_dir, "feature_selector.pkl")
        with open(selector_path, 'wb') as f:
            pickle.dump(self.selector, f)

        # Save column names
        columns_path = os.path.join(preprocessing_dir, "column_names.json")
        columns_info = {
            'plot_col': self.plot_col,
            'genre_col': self.genre_col,
            'location_col': self.location_col
        }
        with open(columns_path, 'w') as f:
            json.dump(columns_info, f, indent=4)

        # Save the entire pipeline
        pipeline_path = os.path.join(preprocessing_dir, "preprocessing_pipeline.pkl")
        with open(pipeline_path, 'wb') as f:
            pickle.dump(self.pipeline, f)

        print(f"Preprocessing components saved to {preprocessing_dir}")

        # Save info about the preprocessing components
        preprocessing_info = {
            'timestamp': timestamp,
            'directory': preprocessing_dir,
            'tfidf_path': tfidf_path,
            'mlb_path': mlb_path,
            'selector_path': selector_path,
            'columns_path': columns_path,
            'pipeline_path': pipeline_path
        }

        info_path = os.path.join(preprocessing_dir, "preprocessing_info.json")
        with open(info_path, 'w') as f:
            json.dump(preprocessing_info, f, indent=4)

        return preprocessing_info

    @classmethod
    def load(cls, models_dir, preprocessing_dir=None, models_save_dir=None):
        """
        Load a GenrePredictor with trained models and preprocessing components.

        Parameters:
        -----------
        models_dir : str
            Directory containing saved models
        preprocessing_dir : str, default=None
            Directory containing preprocessing components. If None, will look for the most recent.
        models_save_dir : str, default=None
            Directory to save new models. If None, uses the same as models_dir.

        Returns:
        --------
        GenrePredictor : Loaded predictor with models and preprocessing components
        """
        # Create a new instance
        predictor = cls(models_dir=models_save_dir or models_dir)

        # Find preprocessing directory if not specified
        if preprocessing_dir is None:
            preprocessing_dirs = []
            for dirname in os.listdir(models_dir):
                if 'preprocessing' in dirname and os.path.isdir(os.path.join(models_dir, dirname)):
                    preprocessing_dirs.append(os.path.join(models_dir, dirname))

            if not preprocessing_dirs:
                raise ValueError(f"No preprocessing directories found in {models_dir}")

            # Use the most recent preprocessing directory
            preprocessing_dirs.sort(reverse=True)
            preprocessing_dir = preprocessing_dirs[0]

        # Load preprocessing components
        predictor._load_preprocessing_components(preprocessing_dir)

        # Find models directory
        saver = ModelSaver(base_dir=models_dir)
        saved_dirs = saver.find_saved_models()

        if not saved_dirs:
            print(f"No saved models found in {models_dir}")
            return predictor

        # Load models from the most recent directory
        models_dir = saved_dirs[0]['directory']
        predictor.evaluator = saver.load_models_into_evaluator(models_dir)

        return predictor

    def _load_preprocessing_components(self, preprocessing_dir):
        """
        Load preprocessing components from disk.

        Parameters:
        -----------
        preprocessing_dir : str
            Directory containing preprocessing components
        """
        # Check if preprocessing_info.json exists
        info_path = os.path.join(preprocessing_dir, "preprocessing_info.json")
        if os.path.exists(info_path):
            with open(info_path, 'r') as f:
                preprocessing_info = json.load(f)

            # Load TF-IDF vectorizer
            with open(preprocessing_info['tfidf_path'], 'rb') as f:
                self.tfidf = pickle.load(f)

            # Load MultiLabelBinarizer
            with open(preprocessing_info['mlb_path'], 'rb') as f:
                self.mlb = pickle.load(f)

            # Load SelectKBest
            with open(preprocessing_info['selector_path'], 'rb') as f:
                self.selector = pickle.load(f)

            # Load column names
            with open(preprocessing_info['columns_path'], 'r') as f:
                columns_info = json.load(f)
                self.plot_col = columns_info['plot_col']
                self.genre_col = columns_info['genre_col']
                self.location_col = columns_info['location_col']

            # Load the entire pipeline
            with open(preprocessing_info['pipeline_path'], 'rb') as f:
                self.pipeline = pickle.load(f)
        else:
            # Try to load individual components
            tfidf_path = os.path.join(preprocessing_dir, "tfidf_vectorizer.pkl")
            mlb_path = os.path.join(preprocessing_dir, "multilabel_binarizer.pkl")
            selector_path = os.path.join(preprocessing_dir, "feature_selector.pkl")
            columns_path = os.path.join(preprocessing_dir, "column_names.json")
            pipeline_path = os.path.join(preprocessing_dir, "preprocessing_pipeline.pkl")

            if os.path.exists(tfidf_path):
                with open(tfidf_path, 'rb') as f:
                    self.tfidf = pickle.load(f)

            if os.path.exists(mlb_path):
                with open(mlb_path, 'rb') as f:
                    self.mlb = pickle.load(f)

            if os.path.exists(selector_path):
                with open(selector_path, 'rb') as f:
                    self.selector = pickle.load(f)

            if os.path.exists(columns_path):
                with open(columns_path, 'r') as f:
                    columns_info = json.load(f)
                    self.plot_col = columns_info['plot_col']
                    self.genre_col = columns_info['genre_col']
                    self.location_col = columns_info['location_col']

            if os.path.exists(pipeline_path):
                with open(pipeline_path, 'rb') as f:
                    self.pipeline = pickle.load(f)

        print(f"Preprocessing components loaded from {preprocessing_dir}")

    def predict(self, data, threshold=0.5):
        """
        Predict genres for new movie data.

        Parameters:
        -----------
        data : DataFrame or dict
            Movie data containing plot and location columns
        threshold : float, default=0.5
            Probability threshold for positive predictions

        Returns:
        --------
        list : Predicted genres for each movie
        """
        if self.pipeline is None or self.evaluator.best_model is None:
            raise ValueError("Model not trained or loaded. Train models or load a saved model first.")

        # Convert dict to DataFrame if needed
        if isinstance(data, dict):
            data = pd.DataFrame([data])

        # Ensure required columns are present
        required_cols = [self.plot_col, self.location_col]
        for col in required_cols:
            if col not in data.columns:
                raise ValueError(f"Required column '{col}' not found in input data")

        # Transform data using the pipeline
        X_transformed = self.pipeline.transform(data)

        # Make probability predictions
        try:
            # Try to use predict_proba if available
            pred_proba = self.evaluator.best_model.predict_proba(X_transformed)
            # Apply threshold to probability predictions
            predictions = (pred_proba >= threshold).astype(int)
        except (AttributeError, NotImplementedError):
            # Fall back to regular predict if predict_proba is not available
            predictions = self.evaluator.best_model.predict(X_transformed)

        # Convert binary predictions to genre labels
        predicted_genres = []
        for pred in predictions:
            genres = [self.mlb.classes_[i] for i in range(len(pred)) if pred[i]]
            predicted_genres.append(genres)

        return predicted_genres
