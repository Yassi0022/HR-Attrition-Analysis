"""Shared utilities for the attrition analysis pipeline."""

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd

from attrition.config import FIGURE_DPI, FIGURE_FORMAT, IMG_DIR, MODELS_DIR


def ensure_dir(path: Path) -> Path:
    """Create directory if it doesn't exist."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_figure(
    fig: plt.Figure,
    filename: str,
    directory: Path | None = None,
    dpi: int = FIGURE_DPI,
    fmt: str = FIGURE_FORMAT,
    **kwargs,
) -> Path:
    """Save matplotlib figure to file and close it."""
    if directory is None:
        directory = IMG_DIR
    ensure_dir(directory)
    filepath = directory / f"{filename}.{fmt}"
    fig.savefig(filepath, dpi=dpi, bbox_inches="tight", **kwargs)
    plt.close(fig)
    return filepath


def load_data(path: Path | None = None) -> pd.DataFrame:
    """Load the raw attrition dataset."""
    if path is None:
        from attrition.config import RAW_DATA_PATH

        path = RAW_DATA_PATH
    return pd.read_csv(path)


def save_model(model, filename: str, directory: Path | None = None) -> Path:
    """Save a fitted model using joblib."""
    if directory is None:
        directory = MODELS_DIR
    ensure_dir(directory)
    filepath = directory / f"{filename}.joblib"
    joblib.dump(model, filepath)
    return filepath


def load_model(filename: str, directory: Path | None = None):
    """Load a fitted model using joblib."""
    if directory is None:
        directory = MODELS_DIR
    filepath = directory / f"{filename}.joblib"
    return joblib.load(filepath)


def save_dataframe(df: pd.DataFrame, filename: str, directory: Path | None = None) -> Path:
    """Save DataFrame to CSV."""
    if directory is None:
        directory = IMG_DIR.parent / "reports"
    ensure_dir(directory)
    filepath = directory / f"{filename}.csv"
    df.to_csv(filepath, index=False)
    return filepath
