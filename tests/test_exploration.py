"""Tests for exploration module."""

import pandas as pd

from attrition.exploration import print_basic_stats, run_exploration
from attrition.utils import load_data


def test_run_exploration_returns_dataframe():
    """Test that run_exploration returns a DataFrame with correct structure."""
    df = load_data()
    result = run_exploration(df)

    assert isinstance(result, pd.DataFrame)
    assert "Department" in result.columns
    assert "Attrition_Percentage" in result.columns
    assert len(result) > 0


def test_run_exploration_sorted_descending():
    """Test that results are sorted by attrition percentage descending."""
    df = load_data()
    result = run_exploration(df)

    attrition_pct = result["Attrition_Percentage"].values
    assert list(attrition_pct) == sorted(attrition_pct, reverse=True)


def test_run_exploration_all_departments():
    """Test that all departments are included."""
    df = load_data()
    result = run_exploration(df)

    departments_in_result = set(result["Department"].values)
    departments_in_data = set(df["Department"].unique())

    assert departments_in_result == departments_in_data


def test_print_basic_stats(capsys):
    """Test that print_basic_stats outputs expected information."""
    df = load_data()
    print_basic_stats(df)

    captured = capsys.readouterr()
    output = captured.out

    assert "Dataset Overview" in output
    assert "Shape:" in output
    assert "Columns" in output
    assert "Data Types:" in output
    assert "Missing Values:" in output
    assert "Target Distribution:" in output
    assert "Attrition" in output
