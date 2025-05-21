# Multi-Model Machine Learning Evaluation System

This project implements a comprehensive system for evaluating multiple machine learning models from basic to advanced complexity. It's designed to help you find the best performing model for your specific task by comparing various algorithms and their performance metrics.

## Features

- **Multiple Model Support**: Evaluates models from basic (Logistic Regression, Naive Bayes) to advanced (Neural Networks, Deep Learning)
- **Comprehensive Evaluation**: Calculates multiple metrics including F1-score, precision, recall, and hamming loss
- **Visualization Tools**: Generates comparison plots, confusion matrices, and feature importance visualizations
- **Hyperparameter Tuning**: Includes tools for optimizing model parameters
- **Ensemble Methods**: Combines multiple models for improved performance
- **Flexible API**: Easy to extend with custom models and evaluation metrics

## Project Structure

- `model_evaluator.py`: The main class that implements the model evaluation framework
- `model_evaluation_demo.ipynb`: Jupyter notebook demonstrating how to use the system
- `MLPRO.ipynb`: Original notebook with data preprocessing and initial model implementation
- `wiki_movie_plots_deduped_cleaned.csv`: Preprocessed dataset for movie genre prediction

## Installation

1. Clone this repository
2. Create a virtual environment:
   ```
   python -m venv ml_env
   ```
3. Activate the environment:
   - Windows: `ml_env\Scripts\activate`
   - Linux/Mac: `source ml_env/bin/activate`
4. Install required packages:
   ```
   pip install pandas numpy matplotlib seaborn scikit-learn scipy
   ```
5. Optional packages for advanced models:
   ```
   pip install xgboost lightgbm tensorflow
   ```

## Usage

### Basic Usage

```python
from model_evaluator import ModelEvaluator

# Create an evaluator
evaluator = ModelEvaluator()

# Add models to evaluate
evaluator.add_basic_models()
evaluator.add_intermediate_models()
evaluator.add_advanced_models()

# Evaluate all models
evaluator.evaluate_models(X, y)

# Print summary of results
evaluator.print_summary()

# Visualize model comparison
evaluator.plot_model_comparison()
```

### Advanced Usage

```python
# Tune hyperparameters for a specific model
param_grid = {
    'estimator__n_estimators': [50, 100, 200],
    'estimator__max_depth': [None, 10, 20, 30]
}
evaluator.tune_hyperparameters('Random Forest', param_grid, X, y)

# Create an ensemble of top models
top_models = ['Random Forest', 'XGBoost', 'Neural Network MLP']
ensemble_pred = evaluator.ensemble_predictions(X_test, models_to_use=top_models)
```

## Example: Movie Genre Prediction

The included demo notebook shows how to use the system to predict movie genres based on plot descriptions. It demonstrates:

1. Data preprocessing and feature engineering
2. Model evaluation and comparison
3. Visualization of results
4. Hyperparameter tuning
5. Ensemble prediction

## Supported Models

### Basic Models
- Logistic Regression
- Multinomial Naive Bayes
- Decision Tree

### Intermediate Models
- Random Forest
- Linear SVM
- Gradient Boosting
- XGBoost (if installed)
- LightGBM (if installed)

### Advanced Models
- Neural Network MLP
- Deep Learning with TensorFlow (if installed)

## Extending the System

You can add custom models to the evaluator:

```python
from sklearn.ensemble import AdaBoostClassifier
from sklearn.multiclass import OneVsRestClassifier

custom_model = OneVsRestClassifier(AdaBoostClassifier())
evaluator.add_custom_model('AdaBoost', custom_model)
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- The movie dataset is derived from "Wiki Movie Plots" available on Kaggle
- This project was created as a demonstration of machine learning model evaluation techniques
