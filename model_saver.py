import os
import pickle
import json
import datetime
import pandas as pd
import numpy as np

class ModelSaver:
    """
    A utility class to save trained machine learning models and their metadata.

    This class provides functionality to:
    - Save trained models to disk using pickle
    - Save model performance metrics
    - Create organized directory structures for model storage
    - Load saved models for later use
    """

    def __init__(self, base_dir='saved_models'):
        """
        Initialize the ModelSaver.

        Parameters:
        -----------
        base_dir : str, default='saved_models'
            Base directory where models will be saved
        """
        self.base_dir = base_dir

        # Create the base directory if it doesn't exist
        if not os.path.exists(base_dir):
            os.makedirs(base_dir)

    def save_all_models(self, evaluator, dataset_name=None, include_data=False, X=None, y=None):
        """
        Save all models from a ModelEvaluator instance.

        Parameters:
        -----------
        evaluator : ModelEvaluator
            The ModelEvaluator instance containing trained models
        dataset_name : str, default=None
            Name of the dataset used for training (used in directory naming)
        include_data : bool, default=False
            Whether to save the training/test data along with the models
        X : array-like, default=None
            Features used for training (only saved if include_data=True)
        y : array-like, default=None
            Target labels used for training (only saved if include_data=True)

        Returns:
        --------
        dict : Information about saved models including paths
        """
        if not evaluator.results:
            print("No models have been evaluated yet. Nothing to save.")
            return {}

        # Create timestamp for this save operation
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        # Create directory name
        if dataset_name:
            save_dir = os.path.join(self.base_dir, f"{dataset_name}_{timestamp}")
        else:
            save_dir = os.path.join(self.base_dir, f"models_{timestamp}")

        # Create the directory
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # Dictionary to store information about saved models
        saved_info = {
            'timestamp': timestamp,
            'base_directory': save_dir,
            'models': {}
        }

        # Save each model
        for model_name, result in evaluator.results.items():
            # Create a safe filename
            safe_name = model_name.replace(' ', '_').lower()
            model_dir = os.path.join(save_dir, safe_name)

            # Create model-specific directory
            if not os.path.exists(model_dir):
                os.makedirs(model_dir)

            # Save the model
            model_path = os.path.join(model_dir, f"{safe_name}_model.pkl")
            with open(model_path, 'wb') as f:
                pickle.dump(result['model'], f)

            # Save model metrics
            metrics = {
                'f1_micro': result['f1_micro'],
                'f1_macro': result['f1_macro'],
                'f1_weighted': result['f1_weighted'],
                'hamming_loss': result['hamming_loss'],
                'is_best_model': (model_name == evaluator.best_model_name)
            }

            metrics_path = os.path.join(model_dir, f"{safe_name}_metrics.json")
            with open(metrics_path, 'w') as f:
                json.dump(metrics, f, indent=4)

            # Save detailed classification report
            report_path = os.path.join(model_dir, f"{safe_name}_report.json")

            # Convert numpy values to Python native types for JSON serialization
            report_dict = self._convert_report_for_json(result['report'])

            with open(report_path, 'w') as f:
                json.dump(report_dict, f, indent=4)

            # Store information about this model
            saved_info['models'][model_name] = {
                'directory': model_dir,
                'model_path': model_path,
                'metrics_path': metrics_path,
                'report_path': report_path,
                'metrics': metrics
            }

        # Save overall summary
        summary_df = evaluator.print_summary()
        summary_path = os.path.join(save_dir, "model_summary.csv")
        summary_df.to_csv(summary_path, index=False)
        saved_info['summary_path'] = summary_path

        # Save feature names and target names if available
        if evaluator.feature_names is not None:
            feature_path = os.path.join(save_dir, "feature_names.json")
            with open(feature_path, 'w') as f:
                json.dump(evaluator.feature_names, f, indent=4)
            saved_info['feature_names_path'] = feature_path

        if evaluator.target_names is not None:
            target_path = os.path.join(save_dir, "target_names.json")
            with open(target_path, 'w') as f:
                json.dump(evaluator.target_names.tolist() if isinstance(evaluator.target_names, np.ndarray)
                         else evaluator.target_names, f, indent=4)
            saved_info['target_names_path'] = target_path

        # Save the dataset if requested
        if include_data and X is not None and y is not None:
            data_dir = os.path.join(save_dir, "data")
            if not os.path.exists(data_dir):
                os.makedirs(data_dir)

            X_path = os.path.join(data_dir, "X.pkl")
            y_path = os.path.join(data_dir, "y.pkl")

            with open(X_path, 'wb') as f:
                pickle.dump(X, f)

            with open(y_path, 'wb') as f:
                pickle.dump(y, f)

            saved_info['data'] = {
                'X_path': X_path,
                'y_path': y_path
            }

        # Save the saved_info itself for reference
        info_path = os.path.join(save_dir, "save_info.json")

        # Convert any non-serializable objects to strings
        serializable_info = self._make_json_serializable(saved_info)

        with open(info_path, 'w') as f:
            json.dump(serializable_info, f, indent=4)

        print(f"All models saved successfully to {save_dir}")
        return saved_info

    def save_best_model(self, evaluator, dataset_name=None):
        """
        Save only the best model from a ModelEvaluator instance.

        Parameters:
        -----------
        evaluator : ModelEvaluator
            The ModelEvaluator instance containing trained models
        dataset_name : str, default=None
            Name of the dataset used for training (used in directory naming)

        Returns:
        --------
        dict : Information about the saved model including path
        """
        if not evaluator.results or not evaluator.best_model_name:
            print("No models have been evaluated yet or no best model found. Nothing to save.")
            return {}

        # Create timestamp for this save operation
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        # Create directory name
        if dataset_name:
            save_dir = os.path.join(self.base_dir, f"{dataset_name}_best_{timestamp}")
        else:
            save_dir = os.path.join(self.base_dir, f"best_model_{timestamp}")

        # Create the directory
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # Get the best model
        best_model_name = evaluator.best_model_name
        best_model = evaluator.best_model
        best_result = evaluator.results[best_model_name]

        # Create a safe filename
        safe_name = best_model_name.replace(' ', '_').lower()

        # Save the model
        model_path = os.path.join(save_dir, f"{safe_name}_model.pkl")
        with open(model_path, 'wb') as f:
            pickle.dump(best_model, f)

        # Save model metrics
        metrics = {
            'f1_micro': best_result['f1_micro'],
            'f1_macro': best_result['f1_macro'],
            'f1_weighted': best_result['f1_weighted'],
            'hamming_loss': best_result['hamming_loss']
        }

        metrics_path = os.path.join(save_dir, f"{safe_name}_metrics.json")
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=4)

        # Save detailed classification report
        report_path = os.path.join(save_dir, f"{safe_name}_report.json")

        # Convert numpy values to Python native types for JSON serialization
        report_dict = self._convert_report_for_json(best_result['report'])

        with open(report_path, 'w') as f:
            json.dump(report_dict, f, indent=4)

        # Save feature names and target names if available
        if evaluator.feature_names is not None:
            feature_path = os.path.join(save_dir, "feature_names.json")
            with open(feature_path, 'w') as f:
                json.dump(evaluator.feature_names, f, indent=4)

        if evaluator.target_names is not None:
            target_path = os.path.join(save_dir, "target_names.json")
            with open(target_path, 'w') as f:
                json.dump(evaluator.target_names.tolist() if isinstance(evaluator.target_names, np.ndarray)
                         else evaluator.target_names, f, indent=4)

        # Information about the saved model
        saved_info = {
            'timestamp': timestamp,
            'directory': save_dir,
            'model_name': best_model_name,
            'model_path': model_path,
            'metrics_path': metrics_path,
            'report_path': report_path,
            'metrics': metrics
        }

        # Save the saved_info itself for reference
        info_path = os.path.join(save_dir, "save_info.json")

        # Convert any non-serializable objects to strings
        serializable_info = self._make_json_serializable(saved_info)

        with open(info_path, 'w') as f:
            json.dump(serializable_info, f, indent=4)

        print(f"Best model ({best_model_name}) saved successfully to {save_dir}")
        return saved_info

    def load_model(self, model_path):
        """
        Load a saved model from disk.

        Parameters:
        -----------
        model_path : str
            Path to the saved model file

        Returns:
        --------
        object : The loaded model
        """
        if not os.path.exists(model_path):
            print(f"Model file not found: {model_path}")
            return None

        with open(model_path, 'rb') as f:
            model = pickle.load(f)

        return model

    def load_models_into_evaluator(self, save_dir, evaluator=None):
        """
        Load all models from a saved directory into a ModelEvaluator instance.

        Parameters:
        -----------
        save_dir : str
            Path to the directory containing saved models
        evaluator : ModelEvaluator, default=None
            ModelEvaluator instance to load models into. If None, a new instance is created.

        Returns:
        --------
        ModelEvaluator : The ModelEvaluator instance with loaded models
        """
        # Check if the directory exists
        if not os.path.exists(save_dir):
            print(f"Directory not found: {save_dir}")
            return None

        # Check if save_info.json exists
        info_path = os.path.join(save_dir, "save_info.json")
        if not os.path.exists(info_path):
            print(f"save_info.json not found in {save_dir}")
            return None

        # Load save_info.json
        with open(info_path, 'r') as f:
            save_info = json.load(f)

        # Create a new ModelEvaluator if none provided
        if evaluator is None:
            # Import here to avoid circular imports
            from model_evaluator import ModelEvaluator
            evaluator = ModelEvaluator()

        # Load feature names if available
        if 'feature_names_path' in save_info and os.path.exists(save_info['feature_names_path']):
            with open(save_info['feature_names_path'], 'r') as f:
                evaluator.feature_names = json.load(f)

        # Load target names if available
        if 'target_names_path' in save_info and os.path.exists(save_info['target_names_path']):
            with open(save_info['target_names_path'], 'r') as f:
                evaluator.target_names = json.load(f)

        # Check if this is a best model directory or a full models directory
        if 'models' in save_info:
            # This is a full models directory
            for model_name, model_info in save_info['models'].items():
                self._load_model_into_evaluator(evaluator, model_name, model_info)

            # Find and set the best model
            for model_name, model_info in save_info['models'].items():
                if 'metrics' in model_info and 'is_best_model' in model_info['metrics'] and model_info['metrics']['is_best_model']:
                    evaluator.best_model_name = model_name
                    evaluator.best_model = evaluator.results[model_name]['model']
                    evaluator.best_score = model_info['metrics']['f1_micro']
                    break
        else:
            # This is a best model directory
            model_name = save_info['model_name']
            self._load_model_into_evaluator(evaluator, model_name, save_info)
            evaluator.best_model_name = model_name
            evaluator.best_model = evaluator.results[model_name]['model']
            evaluator.best_score = save_info['metrics']['f1_micro']

        print(f"Successfully loaded models from {save_dir}")
        return evaluator

    def _load_model_into_evaluator(self, evaluator, model_name, model_info):
        """
        Helper method to load a single model into the evaluator.

        Parameters:
        -----------
        evaluator : ModelEvaluator
            ModelEvaluator instance to load the model into
        model_name : str
            Name of the model
        model_info : dict
            Information about the model from save_info.json
        """
        # Load the model
        model_path = model_info['model_path']
        model = self.load_model(model_path)

        # Add the model to the evaluator's models dictionary
        evaluator.models[model_name] = model

        # Load metrics
        metrics = model_info['metrics']

        # Load report if available
        report = None
        if 'report_path' in model_info and os.path.exists(model_info['report_path']):
            with open(model_info['report_path'], 'r') as f:
                report = json.load(f)

        # Create a minimal results entry
        evaluator.results[model_name] = {
            'model': model,
            'f1_micro': metrics['f1_micro'],
            'f1_macro': metrics['f1_macro'],
            'f1_weighted': metrics['f1_weighted'],
            'hamming_loss': metrics['hamming_loss'],
            'report': report,
            # These will be None since we don't have the actual test data
            'y_pred': None,
            'y_test': None
        }

    def find_saved_models(self):
        """
        Find all saved model directories.

        Returns:
        --------
        list : List of dictionaries with information about saved model directories
        """
        if not os.path.exists(self.base_dir):
            print(f"Base directory not found: {self.base_dir}")
            return []

        saved_dirs = []

        # List all subdirectories in the base directory
        for dirname in os.listdir(self.base_dir):
            dir_path = os.path.join(self.base_dir, dirname)

            # Check if it's a directory and contains save_info.json
            if os.path.isdir(dir_path) and os.path.exists(os.path.join(dir_path, "save_info.json")):
                # Load save_info.json
                with open(os.path.join(dir_path, "save_info.json"), 'r') as f:
                    save_info = json.load(f)

                # Add basic information about this saved directory
                info = {
                    'directory': dir_path,
                    'name': dirname,
                    'timestamp': save_info.get('timestamp', 'unknown'),
                    'is_best_only': 'model_name' in save_info,  # Check if it's a best model directory
                    'model_count': len(save_info.get('models', {})) if 'models' in save_info else 1
                }

                saved_dirs.append(info)

        # Sort by timestamp (newest first)
        saved_dirs.sort(key=lambda x: x['timestamp'], reverse=True)

        return saved_dirs

    def _convert_report_for_json(self, report):
        """
        Convert classification report dictionary for JSON serialization.

        Parameters:
        -----------
        report : dict
            Classification report dictionary from sklearn

        Returns:
        --------
        dict : JSON-serializable report dictionary
        """
        result = {}
        for key, value in report.items():
            if isinstance(value, dict):
                result[key] = self._convert_report_for_json(value)
            elif isinstance(value, (np.float32, np.float64)):
                result[key] = float(value)
            elif isinstance(value, (np.int32, np.int64)):
                result[key] = int(value)
            else:
                result[key] = value
        return result

    def _make_json_serializable(self, obj):
        """
        Convert a dictionary with potentially non-serializable values to a JSON-serializable dict.

        Parameters:
        -----------
        obj : dict or list or other
            Object to make JSON-serializable

        Returns:
        --------
        object : JSON-serializable version of the input
        """
        if isinstance(obj, dict):
            return {k: self._make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        elif isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj
