from dataclasses import dataclass
from typing import List, Dict, Any, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBRegressor, plot_importance


@dataclass
class datasets:
    train_X: pd.DataFrame
    valid_X: pd.DataFrame
    test_X: pd.DataFrame
    train_y: pd.Series
    valid_y: pd.Series
    test_y: pd.Series

def preprocess_features(
    train_X: pd.DataFrame,
    valid_X: pd.DataFrame,
    test_X: pd.DataFrame,
    categorical_features: List[str] = None
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, LabelEncoder]]:
    """
    Preprocess features by handling categorical variables and missing values.
    
    Args:
        train_X: Training features
        valid_X: Validation features
        test_X: Test features
        categorical_features: List of categorical feature names. If None, automatically detect.
    
    Returns:
        Processed train, valid, test sets and dictionary of label encoders
    """
    # If categorical features not specified, detect them
    if categorical_features is None:
        categorical_features = train_X.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # Initialize dictionary to store label encoders
    label_encoders = {}
    
    # Process each categorical feature
    for feature in categorical_features:
        if feature in train_X.columns:
            # Create and fit label encoder
            le = LabelEncoder()
            # Combine all unique values from train, valid, test
            all_values = pd.concat([
                train_X[feature],
                valid_X[feature],
                test_X[feature]
            ]).unique()
            le.fit(all_values)
            
            # Transform the data
            train_X[feature] = le.transform(train_X[feature])
            valid_X[feature] = le.transform(valid_X[feature])
            test_X[feature] = le.transform(test_X[feature])
            
            # Store the encoder
            label_encoders[feature] = le
    
    # Handle missing values
    train_X = train_X.fillna(train_X.mean())
    valid_X = valid_X.fillna(train_X.mean())  # Use training mean
    test_X = test_X.fillna(train_X.mean())  # Use training mean
    
    return train_X, valid_X, test_X, label_encoders

def train_and_eval_model(
    model: XGBRegressor,
    ds: datasets,
    features_to_include: List[str],
    categorical_features: List[str] = None
) -> Tuple[Dict[str, np.ndarray], Dict[str, float], Dict[str, float]]:
    """
    Fits the XGBRegressor and evaluates performance using multiple metrics.

    Args:
        model: XGBRegressor model to train
        ds: datasets object containing train, validation, and test data
        features_to_include: list of features to include in training
        categorical_features: list of categorical feature names

    Returns:
        predictions: dictionary of train, validation, and test predictions
        metrics: dictionary of train, validation, and test metrics
        feature_importance: dictionary of feature importance scores
    """
    # Preprocess features
    train_X, valid_X, test_X, _ = preprocess_features(
        ds.train_X[features_to_include],
        ds.valid_X[features_to_include],
        ds.test_X[features_to_include],
        categorical_features
    )

    # Train model
    model.fit(
        train_X,
        ds.train_y,
        eval_set=[(valid_X, ds.valid_y)],
        eval_metric="rmse",  # Ensure an evaluation metric is set
        verbose=False
    )

    # Make predictions
    train_y_pred = model.predict(train_X)
    valid_y_pred = model.predict(valid_X)
    test_y_pred = model.predict(test_X)

    predictions = {
        "train": train_y_pred,
        "valid": valid_y_pred,
        "test": test_y_pred,
    }

    # Calculate multiple metrics
    metrics = {}
    for split in ["train", "valid", "test"]:
        y_true = getattr(ds, f"{split}_y")
        y_pred = predictions[split]
        
        metrics[f"{split}_rmse"] = np.sqrt(mean_squared_error(y_true, y_pred))
        metrics[f"{split}_mae"] = mean_absolute_error(y_true, y_pred)
        metrics[f"{split}_r2"] = r2_score(y_true, y_pred)

    # Get feature importance
    importance_scores = model.feature_importances_
    feature_importance = dict(zip(features_to_include, importance_scores))

    return predictions, metrics, feature_importance

def run_hparam_search(
    ds: datasets,
    features_to_include: List[str],
    categorical_features: List[str] = None,
    max_depth_options: List[int] = [3, 5, 7, 9],
    learning_rate_options: List[float] = [0.01, 0.05, 0.1, 0.2],
    n_estimators_options: List[int] = [50, 100, 200],
    min_child_weight_options: List[int] = [1, 3, 5],
    subsample_options: List[float] = [0.8, 0.9, 1.0],
    colsample_bytree_options: List[float] = [0.8, 0.9, 1.0],
    gamma_options: List[float] = [0, 0.1, 0.2]
) -> Tuple[Dict[str, float], Dict[str, Any]]:
    """
    Runs an extensive hyperparameter search for XGBRegressor.

    Args:
        ds: datasets object containing train, validation, and test data
        features_to_include: list of features to include in training
        categorical_features: list of categorical feature names
        Various hyperparameter options to try

    Returns:
        best_metrics: dictionary of metrics for best model
        best_params: dictionary of best hyperparameters
    """
    best_metrics = {"valid_rmse": np.inf}
    best_params = None
    
    # Store all results for analysis
    results = []

    # Iterate over hyperparameter combinations
    for max_depth in max_depth_options:
        for learning_rate in learning_rate_options:
            for n_estimators in n_estimators_options:
                for min_child_weight in min_child_weight_options:
                    for subsample in subsample_options:
                        for colsample_bytree in colsample_bytree_options:
                            for gamma in gamma_options:
                                model_params = {
                                    "max_depth": max_depth,
                                    "learning_rate": learning_rate,
                                    "n_estimators": n_estimators,
                                    "min_child_weight": min_child_weight,
                                    "subsample": subsample,
                                    "colsample_bytree": colsample_bytree,
                                    "gamma": gamma,
                                    "random_state": 0,
                                    "enable_categorical": True if categorical_features else False
                                }

                                # Create and train model
                                model = XGBRegressor(**model_params)
                                _, metrics, _ = train_and_eval_model(
                                    model=model,
                                    ds=ds,
                                    features_to_include=features_to_include,
                                    categorical_features=categorical_features
                                )

                                # Store results
                                results.append({
                                    **model_params,
                                    "valid_rmse": metrics["valid_rmse"],
                                    "valid_r2": metrics["valid_r2"]
                                })

                                # Update best model if validation RMSE improved
                                if metrics["valid_rmse"] < best_metrics["valid_rmse"]:
                                    best_metrics = metrics
                                    best_params = model_params

    # Convert results to DataFrame for analysis
    results_df = pd.DataFrame(results)
    
    # Plot hyperparameter effects
    plt.figure(figsize=(15, 10))
    for i, param in enumerate(model_params.keys()):
        if param != "random_state" and param != "enable_categorical":
            plt.subplot(3, 3, i+1)
            sns.scatterplot(data=results_df, x=param, y="valid_rmse")
            plt.title(f"Effect of {param}")
    plt.tight_layout()
    plt.show()

    return best_metrics, best_params

def plot_model_performance(predictions: Dict[str, np.ndarray], ds: datasets):
    """
    Creates visualization of model performance.
    
    Args:
        predictions: Dictionary of predictions for each split
        ds: datasets object containing true values
    """
    plt.figure(figsize=(15, 5))
    
    # Actual vs Predicted plot
    plt.subplot(1, 2, 1)
    for split in ["train", "valid", "test"]:
        y_true = getattr(ds, f"{split}_y")
        y_pred = predictions[split]
        plt.scatter(y_true, y_pred, alpha=0.5, label=split.capitalize())
    
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
    plt.xlabel("Actual Values")
    plt.ylabel("Predicted Values")
    plt.title("Actual vs Predicted")
    plt.legend()
    
    # Residuals plot
    plt.subplot(1, 2, 2)
    for split in ["train", "valid", "test"]:
        y_true = getattr(ds, f"{split}_y")
        y_pred = predictions[split]
        residuals = y_true - y_pred
        plt.scatter(y_pred, residuals, alpha=0.5, label=split.capitalize())
    
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel("Predicted Values")
    plt.ylabel("Residuals")
    plt.title("Residuals Plot")
    plt.legend()
    
    plt.tight_layout()
    plt.show()

def plot_feature_importance(feature_importance: Dict[str, float], top_n: int = 20):
    """
    Plots feature importance scores.
    
    Args:
        feature_importance: Dictionary of feature importance scores
        top_n: Number of top features to show
    """
    # Convert to DataFrame and sort
    importance_df = pd.DataFrame({
        'Feature': list(feature_importance.keys()),
        'Importance': list(feature_importance.values())
    })
    importance_df = importance_df.sort_values('Importance', ascending=False).head(top_n)
    
    # Plot
    plt.figure(figsize=(10, 6))
    sns.barplot(data=importance_df, x='Importance', y='Feature')
    plt.title(f'Top {top_n} Most Important Features')
    plt.xlabel('Feature Importance Score')
    plt.tight_layout()
    plt.show()

def evaluate_linear_regression(
    ds: datasets,
    features_to_include: List[str],
    cv_folds: int = 5,
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[Dict[str, List[float]], Dict[str, float], LinearRegression]:
    """
    Evaluate linear regression model using k-fold cross validation and final test set.
    
    Args:
        ds: datasets object containing train, validation, and test data
        features_to_include: list of features to include in training
        cv_folds: number of cross-validation folds
        test_size: proportion of data to use for test set
        random_state: random seed for reproducibility
    
    Returns:
        cv_history: dictionary containing training and validation metrics for each fold
        test_metrics: dictionary containing final test set metrics
        final_model: trained linear regression model
    """
    # Initialize results storage
    cv_history = {
        'train_rmse': [],
        'train_mae': [],
        'train_r2': [],
        'val_rmse': [],
        'val_mae': [],
        'val_r2': []
    }
    
    # Combine train and validation sets for cross-validation
    X_train_full = pd.concat([ds.train_X[features_to_include], ds.valid_X[features_to_include]])
    y_train_full = pd.concat([ds.train_y, ds.valid_y])
    
    # Preprocess features
    scaler = StandardScaler()
    X_train_full = pd.DataFrame(scaler.fit_transform(X_train_full), columns=features_to_include)
    X_test = pd.DataFrame(scaler.transform(ds.test_X[features_to_include]), columns=features_to_include)
    
    # Perform k-fold cross validation
    from sklearn.model_selection import KFold
    kf = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X_train_full), 1):
        # Split data
        X_train_fold = X_train_full.iloc[train_idx]
        y_train_fold = y_train_full.iloc[train_idx]
        X_val_fold = X_train_full.iloc[val_idx]
        y_val_fold = y_train_full.iloc[val_idx]
        
        # Train model
        model = LinearRegression()
        model.fit(X_train_fold, y_train_fold)
        
        # Make predictions
        train_pred = model.predict(X_train_fold)
        val_pred = model.predict(X_val_fold)
        
        # Calculate metrics
        cv_history['train_rmse'].append(np.sqrt(mean_squared_error(y_train_fold, train_pred)))
        cv_history['train_mae'].append(mean_absolute_error(y_train_fold, train_pred))
        cv_history['train_r2'].append(r2_score(y_train_fold, train_pred))
        
        cv_history['val_rmse'].append(np.sqrt(mean_squared_error(y_val_fold, val_pred)))
        cv_history['val_mae'].append(mean_absolute_error(y_val_fold, val_pred))
        cv_history['val_r2'].append(r2_score(y_val_fold, val_pred))
        
        print(f"Fold {fold}/{cv_folds}:")
        print(f"  Train RMSE: {cv_history['train_rmse'][-1]:.4f}")
        print(f"  Val RMSE: {cv_history['val_rmse'][-1]:.4f}")
    
    # Train final model on all training data
    final_model = LinearRegression()
    final_model.fit(X_train_full, y_train_full)
    
    # Evaluate on test set
    test_pred = final_model.predict(X_test)
    test_metrics = {
        'test_rmse': np.sqrt(mean_squared_error(ds.test_y, test_pred)),
        'test_mae': mean_absolute_error(ds.test_y, test_pred),
        'test_r2': r2_score(ds.test_y, test_pred)
    }
    
    return cv_history, test_metrics, final_model

def plot_linear_regression_results(
    cv_history: Dict[str, List[float]],
    test_metrics: Dict[str, float]
):
    """
    Plot training and validation metrics across folds, and final test metrics.
    
    Args:
        cv_history: dictionary containing training and validation metrics for each fold
        test_metrics: dictionary containing final test set metrics
    """
    # Create figure with subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Plot RMSE
    axes[0].plot(range(1, len(cv_history['train_rmse']) + 1), 
                 cv_history['train_rmse'], 
                 'bo-', label='Training')
    axes[0].plot(range(1, len(cv_history['val_rmse']) + 1), 
                 cv_history['val_rmse'], 
                 'ro-', label='Validation')
    axes[0].axhline(y=test_metrics['test_rmse'], color='g', linestyle='--', 
                    label=f'Test (RMSE={test_metrics["test_rmse"]:.4f})')
    axes[0].set_xlabel('Fold')
    axes[0].set_ylabel('RMSE')
    axes[0].set_title('Root Mean Squared Error')
    axes[0].legend()
    axes[0].grid(True)
    
    # Plot MAE
    axes[1].plot(range(1, len(cv_history['train_mae']) + 1), 
                 cv_history['train_mae'], 
                 'bo-', label='Training')
    axes[1].plot(range(1, len(cv_history['val_mae']) + 1), 
                 cv_history['val_mae'], 
                 'ro-', label='Validation')
    axes[1].axhline(y=test_metrics['test_mae'], color='g', linestyle='--', 
                    label=f'Test (MAE={test_metrics["test_mae"]:.4f})')
    axes[1].set_xlabel('Fold')
    axes[1].set_ylabel('MAE')
    axes[1].set_title('Mean Absolute Error')
    axes[1].legend()
    axes[1].grid(True)
    
    # Plot R²
    axes[2].plot(range(1, len(cv_history['train_r2']) + 1), 
                 cv_history['train_r2'], 
                 'bo-', label='Training')
    axes[2].plot(range(1, len(cv_history['val_r2']) + 1), 
                 cv_history['val_r2'], 
                 'ro-', label='Validation')
    axes[2].axhline(y=test_metrics['test_r2'], color='g', linestyle='--', 
                    label=f'Test (R²={test_metrics["test_r2"]:.4f})')
    axes[2].set_xlabel('Fold')
    axes[2].set_ylabel('R²')
    axes[2].set_title('R-squared Score')
    axes[2].legend()
    axes[2].grid(True)
    
    plt.tight_layout()
    plt.show()
    
    # Print summary statistics
    print("\nSummary Statistics:")
    print("\nTraining Metrics (mean ± std):")
    print(f"RMSE: {np.mean(cv_history['train_rmse']):.4f} ± {np.std(cv_history['train_rmse']):.4f}")
    print(f"MAE: {np.mean(cv_history['train_mae']):.4f} ± {np.std(cv_history['train_mae']):.4f}")
    print(f"R²: {np.mean(cv_history['train_r2']):.4f} ± {np.std(cv_history['train_r2']):.4f}")
    
    print("\nValidation Metrics (mean ± std):")
    print(f"RMSE: {np.mean(cv_history['val_rmse']):.4f} ± {np.std(cv_history['val_rmse']):.4f}")
    print(f"MAE: {np.mean(cv_history['val_mae']):.4f} ± {np.std(cv_history['val_mae']):.4f}")
    print(f"R²: {np.mean(cv_history['val_r2']):.4f} ± {np.std(cv_history['val_r2']):.4f}")
    
    print("\nTest Metrics:")
    print(f"RMSE: {test_metrics['test_rmse']:.4f}")
    print(f"MAE: {test_metrics['test_mae']:.4f}")
    print(f"R²: {test_metrics['test_r2']:.4f}")