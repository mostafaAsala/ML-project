import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.inspection import permutation_importance
from sklearn.feature_selection import SelectFromModel
import warnings

def analyze_feature_importance(evaluator, X, y, feature_names=None, top_n=20, n_repeats=10, random_state=42):
    """
    Analyze feature importance across multiple models.
    
    Parameters:
    -----------
    evaluator : ModelEvaluator
        Trained ModelEvaluator instance with models
    X : array-like
        Features used for training
    y : array-like
        Target labels
    feature_names : list, default=None
        Names of features (if None, will use generic names)
    top_n : int, default=20
        Number of top features to display
    n_repeats : int, default=10
        Number of repeats for permutation importance
    random_state : int, default=42
        Random seed for reproducibility
        
    Returns:
    --------
    dict : Dictionary containing feature importance DataFrames for each model
    """
    if not evaluator.results:
        raise ValueError("No models have been evaluated yet. Run evaluator.evaluate_models() first.")
    
    # Create feature names if not provided
    if feature_names is None:
        if hasattr(X, 'shape'):
            feature_names = [f"Feature {i}" for i in range(X.shape[1])]
        else:
            feature_names = [f"Feature {i}" for i in range(X.iloc[:, 0].shape[0])]
    
    # Dictionary to store importance results
    importance_results = {}
    
    # Analyze each model
    for model_name, result in evaluator.results.items():
        print(f"Analyzing feature importance for {model_name}...")
        model = result['model']
        
        # Try different methods to get feature importance
        importance_df = None
        
        # Method 1: Direct feature importance (if available)
        try:
            importance_df = get_direct_importance(model, feature_names, top_n)
            if importance_df is not None:
                importance_results[f"{model_name}_direct"] = importance_df
                print(f"  ✓ Direct feature importance extracted")
        except Exception as e:
            print(f"  ✗ Direct importance failed: {str(e)}")
        
        # Method 2: Permutation importance
        try:
            importance_df = get_permutation_importance(model, X, y, feature_names, top_n, n_repeats, random_state)
            if importance_df is not None:
                importance_results[f"{model_name}_permutation"] = importance_df
                print(f"  ✓ Permutation importance calculated")
        except Exception as e:
            print(f"  ✗ Permutation importance failed: {str(e)}")
        
        # Method 3: SelectFromModel (for models with coefficients or feature importances)
        try:
            importance_df = get_selectfrommodel_importance(model, X, y, feature_names, top_n)
            if importance_df is not None:
                importance_results[f"{model_name}_selection"] = importance_df
                print(f"  ✓ SelectFromModel importance calculated")
        except Exception as e:
            print(f"  ✗ SelectFromModel failed: {str(e)}")
    
    # Plot combined feature importance
    plot_combined_importance(importance_results, top_n)
    
    return importance_results

def get_direct_importance(model, feature_names, top_n):
    """Extract direct feature importance from model if available."""
    # For models with feature_importances_ attribute (tree-based models)
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        return create_importance_df(importances, feature_names, top_n)
    
    # For OneVsRestClassifier with feature_importances_
    elif hasattr(model, 'estimators_') and hasattr(model.estimators_[0], 'feature_importances_'):
        importances = np.mean([est.feature_importances_ for est in model.estimators_], axis=0)
        return create_importance_df(importances, feature_names, top_n)
    
    # For linear models with coef_ attribute
    elif hasattr(model, 'coef_'):
        if len(model.coef_.shape) > 1:
            importances = np.abs(model.coef_).mean(axis=0)
        else:
            importances = np.abs(model.coef_)
        return create_importance_df(importances, feature_names, top_n)
    
    # For OneVsRestClassifier with coef_
    elif hasattr(model, 'estimators_') and hasattr(model.estimators_[0], 'coef_'):
        importances = np.mean([np.abs(est.coef_) for est in model.estimators_], axis=0)
        if len(importances.shape) > 1:
            importances = importances.mean(axis=0)
        return create_importance_df(importances, feature_names, top_n)
    
    # For XGBoost models
    elif hasattr(model, 'get_booster') and hasattr(model.get_booster(), 'get_score'):
        try:
            importance_dict = model.get_booster().get_score(importance_type='gain')
            if not importance_dict:  # Empty dict
                return None
            features = list(importance_dict.keys())
            importances = list(importance_dict.values())
            return create_importance_df_from_lists(importances, features, top_n)
        except:
            return None
    
    # For LightGBM models
    elif hasattr(model, 'booster_') and hasattr(model.booster_, 'feature_importance'):
        try:
            importances = model.booster_.feature_importance(importance_type='gain')
            return create_importance_df(importances, feature_names, top_n)
        except:
            return None
    
    return None

def get_permutation_importance(model, X, y, feature_names, top_n, n_repeats, random_state):
    """Calculate permutation importance."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        perm_importance = permutation_importance(
            model, X, y, n_repeats=n_repeats, random_state=random_state
        )
    
    importances = perm_importance.importances_mean
    return create_importance_df(importances, feature_names, top_n)

def get_selectfrommodel_importance(model, X, y, feature_names, top_n):
    """Use SelectFromModel to get feature importance."""
    try:
        selector = SelectFromModel(model, prefit=True)
        selector.fit(X, y)  # This shouldn't actually fit, just compute the threshold
        
        # Get the threshold
        threshold = selector.threshold_
        
        # Get feature importances
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        elif hasattr(model, 'coef_'):
            if len(model.coef_.shape) > 1:
                importances = np.abs(model.coef_).mean(axis=0)
            else:
                importances = np.abs(model.coef_)
        else:
            return None
        
        return create_importance_df(importances, feature_names, top_n)
    except:
        return None

def create_importance_df(importances, feature_names, top_n):
    """Create a DataFrame of feature importances."""
    if len(importances) != len(feature_names):
        raise ValueError(f"Length mismatch: {len(importances)} importances vs {len(feature_names)} feature names")
    
    df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    })
    return df.sort_values('Importance', ascending=False).head(top_n)

def create_importance_df_from_lists(importances, features, top_n):
    """Create a DataFrame from separate lists of features and importances."""
    df = pd.DataFrame({
        'Feature': features,
        'Importance': importances
    })
    return df.sort_values('Importance', ascending=False).head(top_n)

def plot_combined_importance(importance_results, top_n=20):
    """Plot combined feature importance across models."""
    if not importance_results:
        print("No feature importance results to plot.")
        return
    
    # Combine all feature importances
    all_features = set()
    for df in importance_results.values():
        all_features.update(df['Feature'].tolist())
    
    # Create a combined DataFrame
    combined_df = pd.DataFrame({'Feature': list(all_features)})
    
    # Add importance from each model
    for model_name, df in importance_results.items():
        # Normalize importances to 0-1 scale for fair comparison
        max_importance = df['Importance'].max()
        if max_importance > 0:  # Avoid division by zero
            normalized_importance = df['Importance'] / max_importance
            importance_dict = dict(zip(df['Feature'], normalized_importance))
            combined_df[model_name] = combined_df['Feature'].map(importance_dict).fillna(0)
    
    # Calculate average importance across models
    importance_cols = [col for col in combined_df.columns if col != 'Feature']
    if importance_cols:
        combined_df['Average_Importance'] = combined_df[importance_cols].mean(axis=1)
        
        # Sort by average importance and get top features
        top_features_df = combined_df.sort_values('Average_Importance', ascending=False).head(top_n)
        
        # Plot
        plt.figure(figsize=(12, 10))
        sns.barplot(x='Average_Importance', y='Feature', data=top_features_df, color='skyblue')
        plt.title(f'Top {top_n} Features by Average Importance Across Models', fontsize=16)
        plt.xlabel('Average Normalized Importance', fontsize=14)
        plt.ylabel('Feature', fontsize=14)
        plt.tight_layout()
        plt.savefig('Images/average_feature_importance.png')
        plt.show()
        
        # Heatmap of feature importance across models
        plt.figure(figsize=(14, 12))
        heatmap_df = top_features_df.set_index('Feature')[importance_cols]
        sns.heatmap(heatmap_df, cmap='viridis', annot=True, fmt='.2f', linewidths=.5)
        plt.title(f'Feature Importance Heatmap Across Models', fontsize=16)
        plt.tight_layout()
        plt.savefig('Images/feature_importance_heatmap.png')
        plt.show()

# Example usage
if __name__ == "__main__":
    from model_evaluator import ModelEvaluator
    from sklearn.datasets import make_multilabel_classification
    
    # Generate sample data
    X, y = make_multilabel_classification(
        n_samples=1000, 
        n_features=20, 
        n_classes=5, 
        n_labels=2,
        random_state=42
    )
    
    # Create feature names
    feature_names = [f"feature_{i}" for i in range(X.shape[1])]
    
    # Create and train evaluator
    evaluator = ModelEvaluator()
    evaluator.add_basic_models()
    evaluator.add_intermediate_models()
    evaluator.evaluate_models(X, y, feature_names=feature_names)
    
    # Analyze feature importance
    importance_results = analyze_feature_importance(
        evaluator, 
        X, 
        y, 
        feature_names=feature_names,
        top_n=15
    )