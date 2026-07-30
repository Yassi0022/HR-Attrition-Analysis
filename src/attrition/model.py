"""Model training and evaluation module for attrition analysis."""

import json
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import typer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    RocCurveDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from attrition.config import (
    LOGISTIC_REGRESSION_MAX_ITER,
    MODELS_DIR,
    N_JOBS,
    RANDOM_STATE,
)
from attrition.preprocessing import prepare_data, prepare_full_dataset
from attrition.utils import ensure_dir, save_figure


def train_logistic_regression(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    max_iter: int = LOGISTIC_REGRESSION_MAX_ITER,
    random_state: int = RANDOM_STATE,
    n_jobs: int = N_JOBS,
    class_weight: str = "balanced",
) -> LogisticRegression:
    """Train a Logistic Regression model with balanced class weights."""
    model = LogisticRegression(
        max_iter=max_iter,
        random_state=random_state,
        n_jobs=n_jobs,
        class_weight=class_weight,
        solver="lbfgs",
    )
    model.fit(X_train, y_train)
    return model


def train_random_forest(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_estimators: int = 200,
    random_state: int = RANDOM_STATE,
    n_jobs: int = N_JOBS,
    class_weight: str = "balanced",
) -> RandomForestClassifier:
    """Train a Random Forest model with balanced class weights."""
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=n_jobs,
        class_weight=class_weight,
    )
    model.fit(X_train, y_train)
    return model


def evaluate_model(
    model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_name: str = "Model",
) -> dict[str, Any]:
    """Evaluate model and return metrics dictionary."""
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "model": model_name,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_pred_proba),
    }

    typer.echo(f"\n{model_name} Evaluation:")
    typer.echo(f"  Accuracy:  {metrics['accuracy']:.4f}")
    typer.echo(f"  Precision: {metrics['precision']:.4f}")
    typer.echo(f"  Recall:    {metrics['recall']:.4f}")
    typer.echo(f"  F1-Score:  {metrics['f1_score']:.4f}")
    typer.echo(f"  ROC-AUC:   {metrics['roc_auc']:.4f}")
    typer.echo(f"\nClassification Report:\n{classification_report(y_test, y_pred, target_names=['No', 'Yes'])}")

    return metrics


def plot_confusion_matrix(
    y_test: pd.Series,
    y_pred: np.ndarray,
    model_name: str = "Model",
) -> None:
    """Plot and save confusion matrix."""
    fig, ax = plt.subplots(figsize=(6, 5))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["No", "Yes"],
        yticklabels=["No", "Yes"],
        ax=ax,
    )
    ax.set_title(f"Confusion Matrix - {model_name}", fontsize=14, fontweight="bold")
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("Actual", fontsize=12)
    plt.tight_layout()
    save_figure(fig, f"confusion_matrix_{model_name.lower().replace(' ', '_')}")


def plot_roc_curve(
    model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_name: str = "Model",
) -> None:
    """Plot and save ROC curve."""
    fig, ax = plt.subplots(figsize=(8, 6))
    RocCurveDisplay.from_estimator(model, X_test, y_test, ax=ax, name=model_name)
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5)
    ax.set_title(f"ROC Curve - {model_name}", fontsize=14, fontweight="bold")
    plt.tight_layout()
    save_figure(fig, f"roc_curve_{model_name.lower().replace(' ', '_')}")


def get_feature_importance(
    model: RandomForestClassifier,
    feature_names: pd.Index,
    top_n: int = 20,
) -> pd.DataFrame:
    """Extract feature importance from Random Forest model."""
    importances = model.feature_importances_
    importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importances,
    }).sort_values("Importance", ascending=False).head(top_n)

    return importance_df


def plot_feature_importance(
    importance_df: pd.DataFrame,
    model_name: str = "Random Forest",
) -> None:
    """Plot and save feature importance."""
    fig, ax = plt.subplots(figsize=(10, 8))
    top_20 = importance_df.head(20).iloc[::-1]  # Reverse for horizontal bar chart
    sns.barplot(data=top_20, x="Importance", y="Feature", ax=ax, palette="viridis")
    ax.set_title(f"Top 20 Feature Importance - {model_name}", fontsize=14, fontweight="bold")
    ax.set_xlabel("Importance", fontsize=12)
    ax.set_ylabel("Feature", fontsize=12)
    plt.tight_layout()
    save_figure(fig, f"feature_importance_{model_name.lower().replace(' ', '_')}")


def plot_logistic_coefficients(
    model: LogisticRegression,
    feature_names: pd.Index,
    top_n: int = 20,
) -> None:
    """Plot logistic regression coefficients."""
    coef_df = pd.DataFrame({
        "Feature": feature_names,
        "Coefficient": model.coef_[0],
    }).sort_values("Coefficient", key=abs, ascending=False).head(top_n)

    fig, ax = plt.subplots(figsize=(10, 8))
    coef_df = coef_df.iloc[::-1]  # Reverse for horizontal bar chart
    colors = ["red" if c < 0 else "blue" for c in coef_df["Coefficient"]]
    sns.barplot(data=coef_df, x="Coefficient", y="Feature", ax=ax, palette=colors)
    ax.set_title("Logistic Regression Coefficients (Top 20 by Magnitude)", fontsize=14, fontweight="bold")
    ax.set_xlabel("Coefficient Value", fontsize=12)
    ax.set_ylabel("Feature", fontsize=12)
    ax.axvline(x=0, color="black", linewidth=0.5)
    plt.tight_layout()
    save_figure(fig, "logistic_regression_coefficients")


def save_model(
    model: Any,
    preprocessor: Any,
    metrics: dict[str, Any],
    model_name: str,
) -> Path:
    """Save model, preprocessor, and metrics to disk."""
    ensure_dir(MODELS_DIR)

    model_path = MODELS_DIR / f"{model_name.lower().replace(' ', '_')}.joblib"
    preprocessor_path = MODELS_DIR / f"{model_name.lower().replace(' ', '_')}_preprocessor.joblib"
    metrics_path = MODELS_DIR / f"{model_name.lower().replace(' ', '_')}_metrics.json"

    joblib.dump(model, model_path)
    joblib.dump(preprocessor, preprocessor_path)

    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2, default=str)

    typer.echo(f"\nModel saved to: {model_path}")
    typer.echo(f"Preprocessor saved to: {preprocessor_path}")
    typer.echo(f"Metrics saved to: {metrics_path}")

    return model_path


def load_model(model_name: str) -> tuple[Any, Any, dict[str, Any]]:
    """Load model, preprocessor, and metrics from disk."""
    model_path = MODELS_DIR / f"{model_name.lower().replace(' ', '_')}.joblib"
    preprocessor_path = MODELS_DIR / f"{model_name.lower().replace(' ', '_')}_preprocessor.joblib"
    metrics_path = MODELS_DIR / f"{model_name.lower().replace(' ', '_')}_metrics.json"

    model = joblib.load(model_path)
    preprocessor = joblib.load(preprocessor_path)

    with open(metrics_path) as f:
        metrics = json.load(f)

    return model, preprocessor, metrics


def run_training_pipeline(
    df: pd.DataFrame | None = None,
    models_to_train: list | None = None,
) -> dict[str, Any]:
    """
    Run the complete training pipeline.

    Args:
        df: Optional DataFrame (loads from default path if None)
        models_to_train: List of model names to train ["logistic_regression", "random_forest"]

    Returns:
        Dictionary with trained models and their metrics
    """
    if models_to_train is None:
        models_to_train = ["logistic_regression", "random_forest"]

    ensure_dir(MODELS_DIR)

    # Prepare data with proper train/test split (no data leakage)
    typer.echo("Preparing data with train/test split...")
    X_train, X_test, y_train, y_test, preprocessor = prepare_data(df)

    results = {}

    # Train Logistic Regression
    if "logistic_regression" in models_to_train:
        typer.echo("\n" + "=" * 50)
        typer.echo("Training Logistic Regression...")
        typer.echo("=" * 50)

        lr_model = train_logistic_regression(X_train, y_train)
        lr_metrics = evaluate_model(lr_model, X_test, y_test, "Logistic Regression")

        # Visualizations
        y_pred_lr = lr_model.predict(X_test)
        plot_confusion_matrix(y_test, y_pred_lr, "Logistic Regression")
        plot_roc_curve(lr_model, X_test, y_test, "Logistic Regression")
        plot_logistic_coefficients(lr_model, X_train.columns)

        save_model(lr_model, preprocessor, lr_metrics, "Logistic Regression")
        results["logistic_regression"] = {"model": lr_model, "metrics": lr_metrics}

    # Train Random Forest
    if "random_forest" in models_to_train:
        typer.echo("\n" + "=" * 50)
        typer.echo("Training Random Forest...")
        typer.echo("=" * 50)

        rf_model = train_random_forest(X_train, y_train)
        rf_metrics = evaluate_model(rf_model, X_test, y_test, "Random Forest")

        # Visualizations
        y_pred_rf = rf_model.predict(X_test)
        plot_confusion_matrix(y_test, y_pred_rf, "Random Forest")
        plot_roc_curve(rf_model, X_test, y_test, "Random Forest")

        # Feature importance
        importance_df = get_feature_importance(rf_model, X_train.columns)
        typer.echo("\nTop 20 Feature Importances:")
        typer.echo(importance_df.to_string(index=False))
        plot_feature_importance(importance_df, "Random Forest")

        save_model(rf_model, preprocessor, rf_metrics, "Random Forest")
        results["random_forest"] = {"model": rf_model, "metrics": rf_metrics}

    typer.echo("\n" + "=" * 50)
    typer.echo("Training Complete!")
    typer.echo("=" * 50)

    return results


def run_feature_importance_analysis(
    df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Run feature importance analysis on the full dataset using Random Forest.
    This is for exploratory analysis, not for final model evaluation.
    """
    typer.echo("Preparing full dataset for feature importance analysis...")
    X, y, preprocessor = prepare_full_dataset(df)

    typer.echo("Training Random Forest on full dataset...")
    rf_model = train_random_forest(X, y)

    importance_df = get_feature_importance(rf_model, X.columns, top_n=30)
    typer.echo("\nTop 30 Feature Importances:")
    typer.echo(importance_df.to_string(index=False))

    plot_feature_importance(importance_df, "Random Forest (Full Dataset)")

    # Save results
    ensure_dir(MODELS_DIR)
    importance_path = MODELS_DIR / "feature_importance_full_dataset.csv"
    importance_df.to_csv(importance_path, index=False)
    typer.echo(f"\nFeature importance saved to: {importance_path}")

    return importance_df


if __name__ == "__main__":
    # Run training pipeline
    results = run_training_pipeline()
