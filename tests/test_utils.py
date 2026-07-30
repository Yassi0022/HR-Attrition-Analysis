"""Tests for utility functions."""

import tempfile
from pathlib import Path

import pandas as pd

from attrition.config import FIGURE_FORMAT
from attrition.utils import (
    ensure_dir,
    load_data,
    load_model,
    save_dataframe,
    save_figure,
    save_model,
)


def test_ensure_dir_creates_directory():
    """Test that ensure_dir creates directory if it doesn't exist."""
    with tempfile.TemporaryDirectory() as tmpdir:
        new_dir = Path(tmpdir) / "new_dir" / "subdir"
        result = ensure_dir(new_dir)
        assert new_dir.exists()
        assert new_dir.is_dir()
        assert result == new_dir


def test_ensure_dir_returns_existing_directory():
    """Test that ensure_dir returns existing directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        existing_dir = Path(tmpdir)
        result = ensure_dir(existing_dir)
        assert result == existing_dir


def test_save_figure():
    """Test saving matplotlib figure."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    ax.plot([1, 2, 3], [1, 2, 3])

    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)
        filepath = save_figure(fig, "test_plot", directory=output_dir)

        assert filepath.exists()
        assert filepath.suffix == f".{FIGURE_FORMAT}"
        assert filepath.name == f"test_plot.{FIGURE_FORMAT}"

        # Figure should be closed after saving
        assert not plt.fignum_exists(fig.number)


def test_save_figure_with_custom_params():
    """Test save_figure with custom DPI and format."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    ax.plot([1, 2, 3], [1, 2, 3])

    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)
        filepath = save_figure(
            fig,
            "test_plot",
            directory=output_dir,
            dpi=300,
            fmt="pdf",
        )

        assert filepath.exists()
        assert filepath.suffix == ".pdf"
        assert filepath.name == "test_plot.pdf"


def test_load_data():
    """Test loading the raw data."""
    df = load_data()
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert "Attrition" in df.columns
    assert len(df) > 1000  # IBM HR dataset has ~1470 rows


def test_load_data_with_custom_path():
    """Test loading data with custom path."""
    # Create a small test CSV
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        test_df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
        test_df.to_csv(f.name, index=False)
        temp_path = Path(f.name)

    try:
        loaded_df = load_data(temp_path)
        assert len(loaded_df) == 3
        assert list(loaded_df.columns) == ["A", "B"]
    finally:
        temp_path.unlink()


def test_save_and_load_model():
    """Test saving and loading a model."""
    from sklearn.linear_model import LogisticRegression

    model = LogisticRegression()
    model.fit([[1, 2], [3, 4], [5, 6]], [0, 1, 0])

    with tempfile.TemporaryDirectory() as tmpdir:
        model_dir = Path(tmpdir)
        model_path = save_model(model, "test_model", directory=model_dir)

        assert model_path.exists()
        assert model_path.suffix == ".joblib"

        loaded_model = load_model("test_model", directory=model_dir)
        assert hasattr(loaded_model, "predict")
        assert loaded_model.predict([[1, 2]])[0] == 0


def test_save_dataframe():
    """Test saving DataFrame to CSV."""
    df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})

    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)
        filepath = save_dataframe(df, "test_data", directory=output_dir)

        assert filepath.exists()
        assert filepath.suffix == ".csv"

        loaded_df = pd.read_csv(filepath)
        pd.testing.assert_frame_equal(loaded_df, df)
