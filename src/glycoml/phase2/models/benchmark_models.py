"""Baselines for lectin-glycan interaction prediction."""

from __future__ import annotations

from typing import Dict, List, Tuple
import warnings

try:
    from sklearn.svm import SVR, SVC
    from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
except ImportError:  # pragma: no cover
    SVR = None
    SVC = None
    RandomForestRegressor = None
    RandomForestClassifier = None


def check_sklearn() -> None:
    if SVR is None or SVC is None:
        raise ImportError("scikit-learn is required for baseline models.")


def train_regression_baselines(x_train, y_train) -> Dict[str, object]:
    check_sklearn()
    models: Dict[str, object] = {}
    models["svr"] = SVR(kernel="rbf")
    models["rf_regressor"] = RandomForestRegressor(n_estimators=200, random_state=13)
    for model in models.values():
        model.fit(x_train, y_train)
    return models


def train_classification_baselines(x_train, y_train) -> Dict[str, object]:
    check_sklearn()
    models: Dict[str, object] = {}
    models["svc"] = SVC(kernel="rbf", probability=True)
    models["rf_classifier"] = RandomForestClassifier(n_estimators=200, random_state=13)
    for model in models.values():
        model.fit(x_train, y_train)
    return models
