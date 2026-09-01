"""Model training and chronological evaluation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .features import FEATURE_COLUMNS


@dataclass(frozen=True)
class ModelReport:
    name: str
    accuracy: float
    log_loss: float
    brier: float
    test_matches: int


def _brier_score(y_true: np.ndarray, probabilities: np.ndarray, classes: np.ndarray) -> float:
    one_hot = (y_true[:, None] == classes[None, :]).astype(float)
    return float(np.mean(np.sum((probabilities - one_hot) ** 2, axis=1)))


def evaluate_models(feature_frame: pd.DataFrame, test_fraction: float = 0.25) -> list[ModelReport]:
    """Train on older matches and evaluate on the newest chronological block."""

    if not 0.1 <= test_fraction <= 0.5:
        raise ValueError("test_fraction must be between 0.1 and 0.5")
    if len(feature_frame) < 80:
        raise ValueError("at least 80 matches are required for evaluation")

    split = int(len(feature_frame) * (1.0 - test_fraction))
    train, test = feature_frame.iloc[:split], feature_frame.iloc[split:]
    x_train = train[FEATURE_COLUMNS]
    y_train = train["result"].to_numpy()
    x_test = test[FEATURE_COLUMNS]
    y_test = test["result"].to_numpy()

    models = {
        "logistic_regression": Pipeline(
            [
                ("scale", StandardScaler()),
                ("model", LogisticRegression(max_iter=1000, random_state=7)),
            ]
        ),
        "gradient_boosting": HistGradientBoostingClassifier(
            max_depth=4, learning_rate=0.07, max_iter=140, random_state=7
        ),
    }
    reports: list[ModelReport] = []
    for name, model in models.items():
        model.fit(x_train, y_train)
        probabilities = model.predict_proba(x_test)
        predictions = model.classes_[np.argmax(probabilities, axis=1)]
        reports.append(
            ModelReport(
                name=name,
                accuracy=float(accuracy_score(y_test, predictions)),
                log_loss=float(log_loss(y_test, probabilities, labels=model.classes_)),
                brier=_brier_score(y_test, probabilities, model.classes_),
                test_matches=len(test),
            )
        )
    return reports
