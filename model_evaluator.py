import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score, hamming_loss
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve, average_precision_score
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.multiclass import OneVsRestClassifier

# Basic Models
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier

# Intermediate Models
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import LinearSVC

# Advanced Models
from sklearn.neural_network import MLPClassifier

# Optional imports for more advanced models (if installed)
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Dropout, Input
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False


class ModelEvaluator:
    """
    A class to evaluate multiple machine learning models for multi-label classification.
    Implements models from basic to advanced and provides comprehensive evaluation metrics.
    """

    def __init__(self, random_state=42, n_jobs=-1):
        """
        Initialize the ModelEvaluator.

        Parameters:
        -----------
        random_state : int, default=42
            Random seed for reproducibility
        n_jobs : int, default=-1
            Number of CPU cores to use for parallel processing (-1 means all cores)
        """
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.models = {}
        self.results = {}
        self.best_model = None
        self.best_score = 0
        self.best_model_name = None
        self.feature_names = None
        self.target_names = None

    def add_basic_models(self):
        """Add basic machine learning models to the evaluator."""
        self.models['Logistic Regression'] = OneVsRestClassifier(
            LogisticRegression(random_state=self.random_state, max_iter=1000, n_jobs=self.n_jobs)
        )

        self.models['Multinomial Naive Bayes'] = OneVsRestClassifier(
            MultinomialNB()
        )

        self.models['Decision Tree'] = OneVsRestClassifier(
            DecisionTreeClassifier(random_state=self.random_state)
        )

        return self

    def add_intermediate_models(self):
        """Add intermediate complexity machine learning models to the evaluator."""
        self.models['Random Forest'] = OneVsRestClassifier(
            RandomForestClassifier(random_state=self.random_state, n_jobs=self.n_jobs)
        )

        self.models['Linear SVM'] = OneVsRestClassifier(
            LinearSVC(random_state=self.random_state, dual=False)
        )

        self.models['Gradient Boosting'] = OneVsRestClassifier(
            GradientBoostingClassifier(random_state=self.random_state)
        )

        if XGBOOST_AVAILABLE:
            self.models['XGBoost'] = OneVsRestClassifier(
                xgb.XGBClassifier(random_state=self.random_state, n_jobs=self.n_jobs)
            )

        if LIGHTGBM_AVAILABLE:
            self.models['LightGBM'] = OneVsRestClassifier(
                lgb.LGBMClassifier(random_state=self.random_state, n_jobs=self.n_jobs)
            )

        return self

    def add_advanced_models(self):
        """Add advanced machine learning models to the evaluator."""
        self.models['Neural Network MLP'] = OneVsRestClassifier(
            MLPClassifier(random_state=self.random_state, max_iter=300, early_stopping=True)
        )

        # Add deep learning model if TensorFlow is available
        if TENSORFLOW_AVAILABLE:
            self.models['Deep Learning'] = self._create_keras_model

        return self

    def add_custom_model(self, name, model):
        """
        Add a custom model to the evaluator.

        Parameters:
        -----------
        name : str
            Name of the model
        model : estimator
            Scikit-learn compatible estimator
        """
        self.models[name] = model
        return self

    def _create_keras_model(self, X_train, y_train, X_test, y_test):
        """
        Create and train a Keras deep learning model.

        This is a helper method used when TensorFlow is available.
        """
        input_dim = X_train.shape[1]
        output_dim = y_train.shape[1]

        # Create a simple neural network
        model = Sequential([
            Dense(256, activation='relu', input_shape=(input_dim,)),
            Dropout(0.3),
            Dense(128, activation='relu'),
            Dropout(0.3),
            Dense(output_dim, activation='sigmoid')
        ])

        # Compile the model
        model.compile(
            loss='binary_crossentropy',
            optimizer=Adam(learning_rate=0.001),
            metrics=['accuracy']
        )

        # Early stopping to prevent overfitting
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True
        )

        # Train the model
        model.fit(
            X_train, y_train,
            epochs=30,
            batch_size=64,
            validation_data=(X_test, y_test),
            callbacks=[early_stopping],
            verbose=0
        )

        # Return the trained model
        return model

    def evaluate_models(self, X, y, test_size=0.2, feature_names=None, target_names=None):
        """
        Evaluate all added models on the given dataset.

        Parameters:
        -----------
        X : array-like
            Features
        y : array-like
            Target labels (multi-label format)
        test_size : float, default=0.2
            Proportion of the dataset to include in the test split
        feature_names : list, default=None
            Names of the features
        target_names : list, default=None
            Names of the target classes
        """
        self.feature_names = feature_names
        self.target_names = target_names

        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state
        )

        # Evaluate each model
        for name, model in self.models.items():
            print(f"Evaluating {name}...")

            # Handle the special case for Keras model
            if name == 'Deep Learning' and TENSORFLOW_AVAILABLE:
                model = model(X_train, y_train, X_test, y_test)
                y_pred = (model.predict(X_test) > 0.5).astype(int)
            else:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

            # Calculate metrics
            report = classification_report(
                y_test, y_pred,
                target_names=target_names,
                output_dict=True,
                zero_division=0
            )

            # Store results
            self.results[name] = {
                'model': model,
                'report': report,
                'f1_micro': report['micro avg']['f1-score'],
                'f1_macro': report['macro avg']['f1-score'],
                'f1_weighted': report['weighted avg']['f1-score'],
                'hamming_loss': hamming_loss(y_test, y_pred),
                'y_pred': y_pred,
                'y_test': y_test
            }

            # Update best model
            if self.results[name]['f1_micro'] > self.best_score:
                self.best_score = self.results[name]['f1_micro']
                self.best_model = model
                self.best_model_name = name

        print(f"Evaluation complete. Best model: {self.best_model_name} (F1-micro: {self.best_score:.4f})")
        return self

    def print_summary(self):
        """Print a summary of the evaluation results."""
        if not self.results:
            print("No models have been evaluated yet.")
            return

        # Create a DataFrame with the results
        summary = pd.DataFrame({
            'Model': list(self.results.keys()),
            'F1-micro': [r['f1_micro'] for r in self.results.values()],
            'F1-macro': [r['f1_macro'] for r in self.results.values()],
            'F1-weighted': [r['f1_weighted'] for r in self.results.values()],
            'Hamming Loss': [r['hamming_loss'] for r in self.results.values()]
        })

        # Sort by F1-micro score
        summary = summary.sort_values('F1-micro', ascending=False).reset_index(drop=True)

        print("Model Performance Summary:")
        print(summary)
        return summary

    def plot_model_comparison(self, metric='f1_micro', figsize=(12, 6)):
        """
        Plot a comparison of model performance.

        Parameters:
        -----------
        metric : str, default='f1_micro'
            Metric to compare ('f1_micro', 'f1_macro', 'f1_weighted', or 'hamming_loss')
        figsize : tuple, default=(12, 6)
            Figure size
        """
        if not self.results:
            print("No models have been evaluated yet.")
            return

        metric_mapping = {
            'f1_micro': 'F1-micro',
            'f1_macro': 'F1-macro',
            'f1_weighted': 'F1-weighted',
            'hamming_loss': 'Hamming Loss'
        }

        if metric not in metric_mapping:
            raise ValueError(f"Invalid metric: {metric}. Choose from {list(metric_mapping.keys())}")

        # Extract the data
        models = list(self.results.keys())
        scores = [r[metric] for r in self.results.values()]

        # Sort by score (descending, except for hamming loss which is ascending)
        if metric == 'hamming_loss':
            sorted_indices = np.argsort(scores)
        else:
            sorted_indices = np.argsort(scores)[::-1]

        sorted_models = [models[i] for i in sorted_indices]
        sorted_scores = [scores[i] for i in sorted_indices]

        # Create the plot
        plt.figure(figsize=figsize)
        bars = plt.barh(sorted_models, sorted_scores, color='skyblue')
        plt.xlabel(metric_mapping[metric])
        plt.title(f'Model Comparison by {metric_mapping[metric]}')
        plt.grid(axis='x', linestyle='--', alpha=0.7)

        # Add value labels
        for i, bar in enumerate(bars):
            plt.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                    f'{sorted_scores[i]:.4f}',
                    va='center')

        plt.tight_layout()
        plt.show()

    def plot_confusion_matrices(self, model_names=None, figsize=(15, 12)):
        """
        Plot confusion matrices for the specified models.

        Parameters:
        -----------
        model_names : list or None, default=None
            List of model names to plot. If None, plots the best model.
        figsize : tuple, default=(15, 12)
            Figure size
        """
        if not self.results:
            print("No models have been evaluated yet.")
            return

        if model_names is None:
            model_names = [self.best_model_name]
        elif not isinstance(model_names, list):
            model_names = [model_names]

        # Filter to only include models that exist
        model_names = [name for name in model_names if name in self.results]

        if not model_names:
            print("No valid models specified.")
            return

        n_models = len(model_names)
        n_classes = self.results[model_names[0]]['y_test'].shape[1]

        if n_classes > 10:
            print(f"Too many classes ({n_classes}) for confusion matrices. Showing only for the first 10 classes.")
            n_classes = 10

        # Create subplots
        fig, axes = plt.subplots(n_models, n_classes, figsize=figsize)
        if n_models == 1 and n_classes == 1:
            axes = np.array([[axes]])
        elif n_models == 1:
            axes = axes.reshape(1, -1)
        elif n_classes == 1:
            axes = axes.reshape(-1, 1)

        class_names = self.target_names[:n_classes] if self.target_names is not None and len(self.target_names) > 0 else [f'Class {i}' for i in range(n_classes)]

        for i, name in enumerate(model_names):
            y_test = self.results[name]['y_test'][:, :n_classes]
            y_pred = self.results[name]['y_pred'][:, :n_classes]

            for j in range(n_classes):
                cm = confusion_matrix(y_test[:, j], y_pred[:, j])
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i, j], cbar=False)
                axes[i, j].set_title(f'{class_names[j]}')

                if j == 0:
                    axes[i, j].set_ylabel(name)

                if i == n_models - 1:
                    axes[i, j].set_xlabel('Predicted')

                if i == 0:
                    axes[i, j].set_title(f'{class_names[j]}')

        plt.tight_layout()
        plt.suptitle('Confusion Matrices by Class and Model', y=1.02, fontsize=16)
        plt.show()

    def tune_hyperparameters(self, model_name, param_grid, X, y, cv=3, scoring='f1_micro', n_iter=10):
        """
        Tune hyperparameters for a specific model.

        Parameters:
        -----------
        model_name : str
            Name of the model to tune
        param_grid : dict
            Dictionary with parameters names as keys and lists of parameter values
        X : array-like
            Features
        y : array-like
            Target labels
        cv : int, default=3
            Number of cross-validation folds
        scoring : str, default='f1_micro'
            Scoring metric to optimize
        n_iter : int, default=10
            Number of parameter settings sampled (for RandomizedSearchCV)

        Returns:
        --------
        self : object
            Returns self
        """
        if model_name not in self.models:
            print(f"Model '{model_name}' not found.")
            return self

        print(f"Tuning hyperparameters for {model_name}...")

        # Use RandomizedSearchCV for efficiency
        search = RandomizedSearchCV(
            self.models[model_name],
            param_distributions=param_grid,
            n_iter=n_iter,
            cv=cv,
            scoring=scoring,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
            verbose=1
        )

        search.fit(X, y)

        print(f"Best parameters: {search.best_params_}")
        print(f"Best {scoring} score: {search.best_score_:.4f}")

        # Update the model with the best estimator
        self.models[model_name] = search.best_estimator_

        return self

    def plot_roc_curves(self, X_test=None, model_names=None, top_n_classes=5, figsize=(12, 8)):
        """
        Plot ROC curves for the specified models.

        Parameters:
        -----------
        X_test : array-like, default=None
            Test features to use for prediction. If None, uses stored predictions.
        model_names : list or None, default=None
            List of model names to plot. If None, plots the best model.
        top_n_classes : int, default=5
            Number of classes to plot (those with highest AUC)
        figsize : tuple, default=(12, 8)
            Figure size
        """
        if not self.results:
            print("No models have been evaluated yet.")
            return

        if model_names is None:
            model_names = [self.best_model_name]
        elif not isinstance(model_names, list):
            model_names = [model_names]

        # Filter to only include models that exist
        model_names = [name for name in model_names if name in self.results]

        if not model_names:
            print("No valid models specified.")
            return

        plt.figure(figsize=figsize)

        # Use different colors for different models
        colors = plt.cm.tab10.colors

        for i, name in enumerate(model_names):
            y_test = self.results[name]['y_test']

            # If X_test is provided, generate new predictions
            if X_test is not None:
                # For models that return decision function
                if hasattr(self.results[name]['model'], 'decision_function'):
                    try:
                        y_score = self.results[name]['model'].decision_function(X_test)
                    except:
                        # If decision_function fails, use predict_proba if available
                        if hasattr(self.results[name]['model'], 'predict_proba'):
                            y_score = self.results[name]['model'].predict_proba(X_test)
                        else:
                            print(f"Model {name} doesn't support decision_function or predict_proba.")
                            continue
                # For models that return probability estimates
                elif hasattr(self.results[name]['model'], 'predict_proba'):
                    y_score = self.results[name]['model'].predict_proba(X_test)
                # For deep learning models
                elif name == 'Deep Learning' and TENSORFLOW_AVAILABLE:
                    y_score = self.results[name]['model'].predict(X_test)
                else:
                    print(f"Model {name} doesn't support decision_function or predict_proba.")
                    continue
            # Otherwise use stored predictions
            else:
                # Use the stored predictions
                y_score = self.results[name]['y_pred']

            n_classes = y_test.shape[1]

            # Calculate ROC curve and AUC for each class
            fpr = {}
            tpr = {}
            roc_auc = {}

            for j in range(n_classes):
                fpr[j], tpr[j], _ = roc_curve(y_test[:, j], y_score[:, j])
                roc_auc[j] = auc(fpr[j], tpr[j])

            # Select top N classes by AUC
            top_classes = sorted(range(n_classes), key=lambda j: roc_auc[j], reverse=True)[:top_n_classes]

            # Plot ROC curves for top classes
            for j in top_classes:
                class_name = self.target_names[j] if self.target_names else f'Class {j}'
                plt.plot(fpr[j], tpr[j], lw=2,
                         label=f'{name} - {class_name} (AUC = {roc_auc[j]:.2f})',
                         color=colors[i % len(colors)])

        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves')
        plt.legend(loc="lower right")
        plt.grid(linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.show()

    def get_feature_importance(self, model_name=None, top_n=20):
        """
        Get feature importance for a specific model.

        Parameters:
        -----------
        model_name : str or None, default=None
            Name of the model. If None, uses the best model.
        top_n : int, default=20
            Number of top features to return

        Returns:
        --------
        DataFrame : Feature importance scores
        """
        if not self.results:
            print("No models have been evaluated yet.")
            return None

        if model_name is None:
            model_name = self.best_model_name

        if model_name not in self.results:
            print(f"Model '{model_name}' not found.")
            return None

        model = self.results[model_name]['model']

        # Different models store feature importance differently
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        elif hasattr(model, 'coef_'):
            importances = np.abs(model.coef_).mean(axis=0)
        elif hasattr(model, 'estimator') and hasattr(model.estimator, 'coef_'):
            importances = np.abs(model.estimator.coef_).mean(axis=0)
        else:
            print(f"Model {model_name} doesn't provide feature importance.")
            return None

        if self.feature_names is None:
            feature_names = [f'Feature {i}' for i in range(len(importances))]
        else:
            feature_names = self.feature_names

        # Create DataFrame with feature importance
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances
        })

        # Sort by importance and get top N
        importance_df = importance_df.sort_values('Importance', ascending=False).head(top_n)

        return importance_df

    def plot_feature_importance(self, model_name=None, top_n=20, figsize=(10, 8)):
        """
        Plot feature importance for a specific model.

        Parameters:
        -----------
        model_name : str or None, default=None
            Name of the model. If None, uses the best model.
        top_n : int, default=20
            Number of top features to plot
        figsize : tuple, default=(10, 8)
            Figure size
        """
        importance_df = self.get_feature_importance(model_name, top_n)

        if importance_df is None:
            return

        plt.figure(figsize=figsize)
        sns.barplot(x='Importance', y='Feature', data=importance_df, palette='viridis')
        plt.title(f'Top {top_n} Feature Importance for {model_name or self.best_model_name}')
        plt.tight_layout()
        plt.show()

    def ensemble_predictions(self, X, models_to_use=None, weights=None):
        """
        Create an ensemble prediction by averaging predictions from multiple models.

        Parameters:
        -----------
        X : array-like
            Features to predict on
        models_to_use : list or None, default=None
            List of model names to use. If None, uses all evaluated models.
        weights : list or None, default=None
            List of weights for each model. If None, uses equal weights.

        Returns:
        --------
        array : Ensemble predictions
        """
        if not self.results:
            print("No models have been evaluated yet.")
            return None

        if models_to_use is None:
            models_to_use = list(self.results.keys())

        # Filter to only include models that exist
        models_to_use = [name for name in models_to_use if name in self.results]

        if not models_to_use:
            print("No valid models specified.")
            return None

        # Default to equal weights if not provided
        if weights is None:
            weights = [1.0 / len(models_to_use)] * len(models_to_use)
        elif len(weights) != len(models_to_use):
            print("Number of weights must match number of models.")
            return None

        # Normalize weights to sum to 1
        weights = np.array(weights) / sum(weights)

        # Get predictions from each model
        all_predictions = []

        for i, name in enumerate(models_to_use):
            model = self.results[name]['model']

            # Handle different model types
            if hasattr(model, 'predict_proba'):
                pred = model.predict_proba(X)
                all_predictions.append(pred * weights[i])
            elif name == 'Deep Learning' and TENSORFLOW_AVAILABLE:
                pred = model.predict(X)
                all_predictions.append(pred * weights[i])
            else:
                # For models without probability output, use binary predictions
                pred = model.predict(X)
                all_predictions.append(pred * weights[i])

        # Average predictions
        ensemble_pred = sum(all_predictions)

        # Convert to binary predictions
        binary_pred = (ensemble_pred > 0.5).astype(int)

        return binary_pred
