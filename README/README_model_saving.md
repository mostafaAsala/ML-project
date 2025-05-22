# Model Saving Functionality

This document explains how to use the `ModelSaver` class to save and load machine learning models from the `ModelEvaluator`.

## Overview

The `ModelSaver` class provides functionality to:

- Save all trained models from a `ModelEvaluator` instance
- Save only the best-performing model
- Save model performance metrics and classification reports
- Load saved models for later use

## Installation

No additional installation is required beyond the dependencies already used in the project:
- Python 3.6+
- NumPy
- Pandas
- scikit-learn
- Pickle (part of Python's standard library)

## Usage

### Importing the ModelSaver

```python
from model_saver import ModelSaver
```

### Creating a ModelSaver Instance

```python
# Create a ModelSaver with default settings (saves to 'saved_models' directory)
saver = ModelSaver()

# Or specify a custom directory
saver = ModelSaver(base_dir='my_models_directory')
```

### Saving All Models

After training and evaluating models with the `ModelEvaluator`, you can save all models:

```python
# Assuming 'evaluator' is your trained ModelEvaluator instance
save_info = saver.save_all_models(
    evaluator=evaluator,
    dataset_name='my_dataset',  # Optional: used for directory naming
    include_data=False          # Set to True to also save the training data
)

# If you want to save the training data as well
save_info = saver.save_all_models(
    evaluator=evaluator,
    dataset_name='my_dataset',
    include_data=True,
    X=X_selected,  # Your feature matrix
    y=y            # Your target matrix
)
```

### Saving Only the Best Model

If you only want to save the best-performing model:

```python
best_model_info = saver.save_best_model(
    evaluator=evaluator,
    dataset_name='my_dataset'  # Optional: used for directory naming
)
```

### Loading a Single Model

```python
# Load a model using its path
model_path = save_info['models']['Model Name']['model_path']
loaded_model = saver.load_model(model_path)

# Or for the best model
model_path = best_model_info['model_path']
loaded_model = saver.load_model(model_path)

# Now you can use the loaded model for predictions
predictions = loaded_model.predict(new_data)
```

### Loading All Models into a ModelEvaluator

You can load all saved models back into a ModelEvaluator instance:

```python
# Create a ModelSaver instance
saver = ModelSaver()

# Find all saved model directories
saved_dirs = saver.find_saved_models()
print(f"Found {len(saved_dirs)} saved model directories")

# Load models from a specific directory
save_dir = saved_dirs[0]['directory']  # Use the most recent directory
evaluator = saver.load_models_into_evaluator(save_dir)

# Now you can use the evaluator as if you had just trained the models
summary = evaluator.print_summary()
evaluator.plot_model_comparison()

# Get the best model
best_model = evaluator.best_model
predictions = best_model.predict(new_data)
```

## Directory Structure

The `ModelSaver` creates the following directory structure:

```
saved_models/
├── dataset_name_timestamp/
│   ├── model_summary.csv
│   ├── feature_names.json
│   ├── target_names.json
│   ├── save_info.json
│   ├── model1_name/
│   │   ├── model1_name_model.pkl
│   │   ├── model1_name_metrics.json
│   │   └── model1_name_report.json
│   ├── model2_name/
│   │   ├── model2_name_model.pkl
│   │   ├── model2_name_metrics.json
│   │   └── model2_name_report.json
│   └── ...
└── dataset_name_best_timestamp/
    ├── best_model_name_model.pkl
    ├── best_model_name_metrics.json
    ├── best_model_name_report.json
    ├── feature_names.json
    ├── target_names.json
    └── save_info.json
```

## Example

See the `save_models_example.py` script for a complete example of how to use the `ModelSaver` class.

```python
# Run the example script
python save_models_example.py
```

The Jupyter notebook `model_evaluation_demo.ipynb` also includes examples of using the `ModelSaver` class in section 7.

## Notes

- Models are saved using Python's pickle module, which means they can only be loaded in a compatible Python environment.
- The saved models include all the preprocessing steps that were part of the model pipeline.
- Performance metrics and classification reports are saved as JSON files for easy inspection.
- The `save_info.json` file contains metadata about all saved models and can be used to locate specific models later.
