"""Visualization module for attrition analysis."""

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import typer

from attrition.config import DEFAULT_FIGSIZE, IMG_DIR
from attrition.utils import ensure_dir, load_data, save_figure


def plot_age_distribution(df: pd.DataFrame) -> None:
    """Plot age distribution with KDE."""
    fig, ax = plt.subplots(figsize=DEFAULT_FIGSIZE)
    sns.histplot(data=df, x="Age", kde=True, bins=20, ax=ax)
    ax.set_title("Age Distribution with KDE", fontsize=14, fontweight="bold")
    ax.set_xlabel("Age", fontsize=12)
    ax.set_ylabel("Frequency", fontsize=12)
    plt.tight_layout()
    save_figure(fig, "age_distribution")


def plot_monthly_income_boxplot(df: pd.DataFrame) -> None:
    """Plot monthly income boxplot."""
    fig, ax = plt.subplots(figsize=DEFAULT_FIGSIZE)
    sns.boxplot(x=df["MonthlyIncome"], ax=ax)
    ax.set_title("Monthly Income Boxplot", fontsize=14, fontweight="bold")
    ax.set_xlabel("Monthly Income", fontsize=12)
    plt.tight_layout()
    save_figure(fig, "monthly_income_boxplot")


def plot_income_by_attrition(df: pd.DataFrame) -> None:
    """Plot monthly income distribution by attrition status."""
    fig, ax = plt.subplots(figsize=DEFAULT_FIGSIZE)
    sns.boxplot(data=df, x="Attrition", y="MonthlyIncome", ax=ax)
    ax.set_title("Monthly Income by Attrition", fontsize=14, fontweight="bold")
    ax.set_xlabel("Attrition", fontsize=12)
    ax.set_ylabel("Monthly Income", fontsize=12)
    plt.tight_layout()
    save_figure(fig, "income_by_attrition")


def plot_attrition_by_categorical(
    df: pd.DataFrame,
    column: str,
    title: str | None = None,
    figsize: tuple[float, float] | None = None,
) -> None:
    """Plot attrition rate by a categorical column."""
    if figsize is None:
        figsize = DEFAULT_FIGSIZE

    attrition_rate = (
        df.groupby(column)["Attrition"]
        .apply(lambda x: (x == "Yes").mean() * 100)
        .sort_values(ascending=False)
    )

    fig, ax = plt.subplots(figsize=figsize)
    sns.barplot(x=attrition_rate.index, y=attrition_rate.values, ax=ax)
    ax.set_title(title or f"Attrition Rate by {column}", fontsize=14, fontweight="bold")
    ax.set_ylabel("Attrition %", fontsize=12)
    ax.set_xlabel(column, fontsize=12)
    ax.tick_params(axis="x", rotation=30)
    plt.tight_layout()
    save_figure(fig, f"attrition_by_{column.lower()}")


def run_all_visualizations(df: pd.DataFrame = None) -> None:
    """Run all visualizations and save to img/ directory."""
    if df is None:
        df = load_data()

    ensure_dir(IMG_DIR)

    typer.echo("Generating visualizations...")
    plot_age_distribution(df)
    typer.echo("  [OK] Age distribution")
    plot_monthly_income_boxplot(df)
    typer.echo("  [OK] Monthly income boxplot")
    plot_income_by_attrition(df)
    typer.echo("  [OK] Income by attrition")

    # Additional categorical visualizations
    categorical_cols = [
        "Department",
        "JobRole",
        "EducationField",
        "BusinessTravel",
        "MaritalStatus",
        "OverTime",
    ]
    for col in categorical_cols:
        if col in df.columns:
            plot_attrition_by_categorical(df, col)
            typer.echo(f"  [OK] Attrition by {col}")

    typer.echo(f"\nAll visualizations saved to {IMG_DIR}/")


if __name__ == "__main__":
    df = load_data()
    run_all_visualizations(df)
