"""Tests for the visuals module."""

import matplotlib
import pandas as pd
import pytest

matplotlib.use("Agg")  # Non-interactive backend for testing
import matplotlib.pyplot as plt

from attrition.visuals import (
    plot_age_distribution,
    plot_attrition_by_categorical,
    plot_income_by_attrition,
    plot_monthly_income_boxplot,
    run_all_visualizations,
)


@pytest.fixture
def sample_data():
    """Create sample data for testing visualizations."""
    n = 80
    return pd.DataFrame({
        "Age": [25, 30, 35, 40, 45, 50, 55, 60] * (n // 8),
        "MonthlyIncome": [3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000] * (n // 8),
        "Attrition": ["Yes", "No"] * (n // 2),
        "Department": ["Sales", "Research", "HR"] * (n // 3) + ["Sales"] * (n % 3),
        "JobRole": ["Manager", "Developer", "Analyst"] * (n // 3) + ["Manager"] * (n % 3),
        "BusinessTravel": ["Travel_Rarely", "Travel_Frequently", "Non-Travel"] * (n // 3) + ["Travel_Rarely"] * (n % 3),
    })


def test_plot_age_distribution(sample_data):
    """Test age distribution plot creation."""
    plot_age_distribution(sample_data)
    # Should not raise any errors
    plt.close("all")


def test_plot_monthly_income_boxplot(sample_data):
    """Test monthly income boxplot creation."""
    plot_monthly_income_boxplot(sample_data)
    plt.close("all")


def test_plot_income_by_attrition(sample_data):
    """Test income by attrition boxplot creation."""
    plot_income_by_attrition(sample_data)
    plt.close("all")


def test_plot_attrition_by_categorical(sample_data):
    """Test attrition by categorical plot creation."""
    plot_attrition_by_categorical(sample_data, "Department")
    plt.close("all")


def test_run_all_visualizations(sample_data):
    """Test running all visualizations."""
    run_all_visualizations(sample_data)
    plt.close("all")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
