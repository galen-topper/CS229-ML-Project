from dataclasses import dataclass
from typing import List
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor


import itertools
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score


@dataclass
class datasets:
    train_X: pd.DataFrame
    valid_X: pd.DataFrame
    test_X: pd.DataFrame
    train_y: pd.Series
    valid_y: pd.Series
    test_y: pd.Series

def train_and_eval_model(
    model: XGBRegressor, ds: datasets, features_to_include: List[str]
):
    """
    This function fits the XGBRegressor on a subset of features using specified hyperparameters.

    Inputs:
    - model: XGBRegressor model to train
    - ds: datasets object containing train, validation, and test data
    - features_to_include: list of features to include in training

    Outputs:
    - predictions: dictionary of train, validation, and test predictions
    - metrics: dictionary of train, validation, and test RMSE
    """
    model.fit(ds.train_X[features_to_include], ds.train_y)

    train_y_pred = model.predict(ds.train_X[features_to_include])
    valid_y_pred = model.predict(ds.valid_X[features_to_include])
    test_y_pred = model.predict(ds.test_X[features_to_include])

    predictions = {
        "train": train_y_pred,
        "valid": valid_y_pred,
        "test": test_y_pred,
    }

    metrics = {
        "train": np.sqrt(mean_squared_error(ds.train_y, train_y_pred)),
        "valid": np.sqrt(mean_squared_error(ds.valid_y, valid_y_pred)),
        "test": np.sqrt(mean_squared_error(ds.test_y, test_y_pred)),
    }

    return predictions, metrics


def run_hparam_search(
    ds: datasets,
    features_to_include: List[str],
    max_depth_options: List[int] = [1, 2, 3],
    learning_rate_options: List[float] = [0.001, 0.01, 0.1, 0.2, 0.3],
    n_estimators_options: List[int] = [2, 5, 10, 25, 50],
):
    """
    This function runs a hyperparameter search for XGBRegressor and selects the best model based on validation RMSE.

    Inputs: 
    - ds: datasets object containing train, validation, and test data
    - features_to_include: list of features to include in training
    - max_depths: list of max_depths to try
    - learning_rates: list of learning_rates to try
    - n_estimators: list of n_estimators to try
    
    Outputs: 
    - best_metrics: dictionary of train, validation, and test RMSE for best model
    - best_params: dictionary of best hyperparameters
    """
    # Initialize variables to track best model
    best_metrics = {"train": np.inf, "valid": np.inf, "test": np.inf}
    best_params = None

    # Iterate over xgboost parameters
    for max_depth in max_depth_options:
        for learning_rate in learning_rate_options:
            for n_estimators in n_estimators_options:
                model_params = {
                    "max_depth": max_depth,
                    "learning_rate": learning_rate,
                    "n_estimators": n_estimators,
                    "enable_categorical": True,
                }

                model = XGBRegressor(
                    random_state=0,
                    **model_params,
                    enable_categorical=True,
                )

                # Train model
                _, model_metrics = train_and_eval_model(
                    ds=ds, model=model, features_to_include=features_to_include
                )

                # Update best model if validation rmse improved
                if model_metrics["valid"] < best_metrics["valid"]:
                    best_metrics = model_metrics
                    best_params = model_params

    return best_metrics, best_params


def hyperparameter_search_logisitc(X_train_val, y_train_val, param_grid, inner_cv):
    """
    Performs hyperparameter tuning using cross-validation.
    
    Iputs:
        - X_train_val (pd.DataFrame): Training + validation data features.
        - y_train_val (pd.Series): Training + validation data labels.
        - param_grid (dict): Dictionary of hyperparameters to tune.
        - inner_cv (StratifiedKFold): Cross-validation splitter for inner loop.

    Outputs:
        dict: Best hyperparameters and their corresponding validation score.
    """
    param_scores = {param: [] for param in itertools.product(param_grid["C"], param_grid["l1_ratio"])}

    # Inner loop for train-validation split
    for train_index, val_index in inner_cv.split(X_train_val, y_train_val):
        # Split into train and validation
        X_train, X_val = X_train_val.iloc[train_index], X_train_val.iloc[val_index]
        y_train, y_val = y_train_val.iloc[train_index], y_train_val.iloc[val_index]

        # Handle missing values using SimpleImputer
        imputer = SimpleImputer(strategy="median")
        X_train = imputer.fit_transform(X_train)
        X_val = imputer.transform(X_val)

        # Standardize features using StandardScaler
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)

        # Iterate over all hyperparameter combinations
        for C, l1_ratio in itertools.product(param_grid["C"], param_grid["l1_ratio"]):
            # Train the logistic regression model with current hyperparameters
            model = LogisticRegression(
                penalty="elasticnet",
                solver="saga",
                l1_ratio=l1_ratio,
                max_iter=10000,
                C=C,
                class_weight="balanced"  # Adjust for class imbalance
            )
            model.fit(X_train, y_train)

            # Validate on the validation set
            val_score = roc_auc_score(y_val, model.predict_proba(X_val)[:, 1])
            param_scores[(C, l1_ratio)].append(val_score)

    # Calculate mean validation scores for each hyperparameter combination
    mean_param_scores = {param: np.mean(scores) for param, scores in param_scores.items()}
    best_params = max(mean_param_scores, key=mean_param_scores.get)

    return {"best_params": best_params, "best_score": mean_param_scores[best_params]}
