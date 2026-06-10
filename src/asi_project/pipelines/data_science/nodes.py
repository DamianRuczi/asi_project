"""Potok 'data_science' — podział danych, trening, ewaluacja + MLflow.

Każde uruchomienie `kedro run` tworzy jeden run w MLflow z parametrami,
metrykami i artefaktem modelu (system śledzenia eksperymentów).
"""

import logging

import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)

MLFLOW_EXPERIMENT = "hotel-cancellation"
# Baza śledzenia eksperymentów (plik w katalogu projektu).
# Podgląd: mlflow ui --backend-store-uri sqlite:///mlflow.db
MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"


def split_data(
    model_input: pd.DataFrame,
    target_column: str,
    test_size: float,
    random_state: int,
):
    """Podział na zbiór treningowy i testowy (ze stratyfikacją po klasie)."""
    X = model_input.drop(columns=[target_column])
    y = model_input[target_column]
    return train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )


def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    model_params: dict,
    random_state: int,
) -> RandomForestClassifier:
    """Trening klasyfikatora na parametrach z conf/base/parameters.yml."""
    model = RandomForestClassifier(
        **model_params, random_state=random_state, n_jobs=-1
    )
    model.fit(X_train, y_train)
    logger.info("train_model: wytrenowano %s", model.__class__.__name__)
    return model


def evaluate_model(
    model: RandomForestClassifier,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    parameters: dict,
) -> dict:
    """Ewaluacja na zbiorze testowym + zapis eksperymentu do MLflow."""
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "f1": round(f1_score(y_test, y_pred), 4),
        "roc_auc": round(roc_auc_score(y_test, y_proba), 4),
    }

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)
    with mlflow.start_run():
        mlflow.log_params(parameters["model"])
        mlflow.log_param("test_size", parameters["test_size"])
        mlflow.log_param("random_state", parameters["random_state"])
        mlflow.log_param("top_n_countries", parameters["top_n_countries"])
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(model, name="model")

    logger.info("evaluate_model: %s", metrics)
    return metrics
