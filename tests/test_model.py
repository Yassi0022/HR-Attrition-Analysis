"""Tests for the model module."""

import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from attrition.model import (
    evaluate_model,
    get_feature_importance,
    plot_confusion_matrix,
    plot_feature_importance,
    plot_logistic_coefficients,
    plot_roc_curve,
    train_logistic_regression,
    train_random_forest,
)


@pytest.fixture
def sample_training_data():
    """Create sample training data."""
    np.random.seed(42)
    n_train = 200
    n_test = 50
    n_features = 10

    X_train = pd.DataFrame(
        np.random.randn(n_train, n_features), columns=[f"feature_{i}" for i in range(n_features)]
    )
    y_train = pd.Series(np.random.choice([0, 1], n_train, p=[0.8, 0.2]))

    X_test = pd.DataFrame(
        np.random.randn(n_test, n_features), columns=[f"feature_{i}" for i in range(n_features)]
    )
    y_test = pd.Series(np.random.choice([0, 1], n_test, p=[0.8, 0.2]))

    return X_train, X_test, y_train, y_test


def test_train_logistic_regression(sample_training_data):
    """Test Logistic Regression training."""
    X_train, _, y_train, _ = sample_training_data

    model = train_logistic_regression(X_train, y_train)

    assert isinstance(model, LogisticRegression)
    assert hasattr(model, "coef_")
    assert hasattr(model, "intercept_")
    assert model.class_weight == "balanced"


def test_train_random_forest(sample_training_data):
    """Test Random Forest training."""
    X_train, _, y_train, _ = sample_training_data

    model = train_random_forest(X_train, y_train)

    assert isinstance(model, RandomForestClassifier)
    assert hasattr(model, "feature_importances_")
    assert model.class_weight == "balanced"


def test_evaluate_model(sample_training_data):
    """Test model evaluation."""
    X_train, X_test, y_train, y_test = sample_training_data

    model = train_logistic_regression(X_train, y_train)
    metrics = evaluate_model(model, X_test, y_test, "Test Model")

    assert isinstance(metrics, dict)
    assert "accuracy" in metrics
    assert "precision" in metrics
    assert "recall" in metrics
    assert "f1_score" in metrics
    assert "roc_auc" in metrics

    # All metrics should be between 0 and 1
    for key in ["accuracy", "precision", "recall", "f1_score", "roc_auc"]:
        assert 0 <= metrics[key] <= 1


def test_get_feature_importance(sample_training_data):
    """Test feature importance extraction."""
    X_train, _, y_train, _ = sample_training_data

    rf_model = train_random_forest(X_train, y_train)
    importance_df = get_feature_importance(rf_model, X_train.columns, top_n=5)

    assert isinstance(importance_df, pd.DataFrame)
    assert len(importance_df) == 5
    assert "Feature" in importance_df.columns
    assert "Importance" in importance_df.columns

    # Should be sorted by importance descending
    assert importance_df["Importance"].is_monotonic_decreasing


def test_plot_feature_importance(sample_training_data):
    """Test feature importance plotting."""
    X_train, _, y_train, _ = sample_training_data

    rf_model = train_random_forest(X_train, y_train)
    importance_df = get_feature_importance(rf_model, X_train.columns)

    plot_feature_importance(importance_df)
    plt.close("all")


def test_plot_logistic_coefficients(sample_training_data):
    """Test logistic regression coefficient plotting."""
    X_train, _, y_train, _ = sample_training_data

    lr_model = train_logistic_regression(X_train, y_train)
    plot_logistic_coefficients(lr_model, X_train.columns)
    plt.close("all")


def test_plot_confusion_matrix(sample_training_data):
    """Test confusion matrix plotting."""
    _, X_test, _, y_test = sample_training_data
    y_pred = np.random.choice([0, 1], len(y_test))

    plot_confusion_matrix(y_test, y_pred, "Test Model")
    plt.close("all")


def test_plot_roc_curve(sample_training_data):
    """Test ROC curve plotting."""
    X_train, X_test, y_train, y_test = sample_training_data

    model = train_logistic_regression(X_train, y_train)
    plot_roc_curve(model, X_test, y_test, "Test Model")
    plt.close("all")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
