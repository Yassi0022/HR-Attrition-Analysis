"""Tests for the correlations module."""

import matplotlib
import pandas as pd
import pytest

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from attrition.correlations import (
    compute_correlation_matrix,
    find_strong_correlations,
    plot_correlation_heatmap,
    run_correlation_analysis,
)


@pytest.fixture
def sample_numeric_data():
    """Create sample numeric data with known correlations."""
    return pd.DataFrame(
        {
            "A": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] * 5,
            "B": [2, 4, 6, 8, 10, 12, 14, 16, 18, 20] * 5,  # Perfect correlation with A
            "C": [10, 9, 8, 7, 6, 5, 4, 3, 2, 1] * 5,  # Negative correlation with A
            "E": [1, 3, 2, 5, 4, 7, 6, 9, 8, 10] * 5,  # Weak correlation
        }
    )


def test_compute_correlation_matrix(sample_numeric_data):
    """Test correlation matrix computation."""
    corr_matrix = compute_correlation_matrix(sample_numeric_data)

    assert isinstance(corr_matrix, pd.DataFrame)
    assert corr_matrix.shape == (4, 4)
    assert list(corr_matrix.columns) == list(sample_numeric_data.columns)
    assert list(corr_matrix.index) == list(sample_numeric_data.columns)

    # Diagonal should be 1.0
    for col in corr_matrix.columns:
        assert abs(corr_matrix.loc[col, col] - 1.0) < 1e-10


def test_find_strong_correlations(sample_numeric_data):
    """Test finding strong correlations."""
    corr_matrix = compute_correlation_matrix(sample_numeric_data)
    strong_corr = find_strong_correlations(corr_matrix, threshold=0.9)

    assert isinstance(strong_corr, pd.DataFrame)
    assert "Feature_1" in strong_corr.columns
    assert "Feature_2" in strong_corr.columns
    assert "Correlation" in strong_corr.columns

    # A and B should have perfect correlation (1.0)
    assert len(strong_corr) >= 1
    top_corr = strong_corr.iloc[0]
    assert abs(top_corr["Correlation"]) >= 0.9


def test_find_strong_correlations_empty(sample_numeric_data):
    """Test finding strong correlations with threshold > 1.0 (impossible)."""
    corr_matrix = compute_correlation_matrix(sample_numeric_data)

    # With threshold > 1.0, no correlations should be found (max correlation is 1.0)
    strong_corr_strict = find_strong_correlations(corr_matrix, threshold=1.001)
    assert len(strong_corr_strict) == 0
    assert list(strong_corr_strict.columns) == ["Feature_1", "Feature_2", "Correlation"]


def test_plot_correlation_heatmap(sample_numeric_data):
    """Test correlation heatmap plotting."""
    corr_matrix = compute_correlation_matrix(sample_numeric_data)
    plot_correlation_heatmap(corr_matrix)
    plt.close("all")


def test_run_correlation_analysis(sample_numeric_data):
    """Test full correlation analysis pipeline."""
    run_correlation_analysis(sample_numeric_data)
    plt.close("all")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
