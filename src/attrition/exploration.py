"""Exploratory Data Analysis module for attrition analysis."""

import pandas as pd
import typer

from attrition.config import DEFAULT_FIGSIZE, IMG_DIR
from attrition.utils import ensure_dir, load_data, save_figure


def run_exploration(df: pd.DataFrame = None) -> pd.DataFrame:
    """
    Run exploratory data analysis and generate attrition by department visualization.

    Returns:
        DataFrame with attrition percentage by department
    """
    if df is None:
        df = load_data()

    ensure_dir(IMG_DIR)

    # Calculate attrition percentage by department
    attrition_ct = pd.crosstab(df["Department"], df["Attrition"])
    attrition_pct = attrition_ct.apply(lambda row: row["Yes"] / row.sum() * 100, axis=1)

    result = pd.DataFrame({
        "Department": attrition_pct.index,
        "Attrition_Percentage": attrition_pct.values,
    }).sort_values("Attrition_Percentage", ascending=False)

    # Create visualization
    import matplotlib.pyplot as plt
    import seaborn as sns

    fig, ax = plt.subplots(figsize=DEFAULT_FIGSIZE)
    sns.barplot(data=result, x="Department", y="Attrition_Percentage", ax=ax)

    ax.set_title("Attrition Rate by Department", fontsize=14, fontweight="bold")
    ax.set_ylabel("Attrition %", fontsize=12)
    ax.set_xlabel("Department", fontsize=12)
    ax.tick_params(axis="x", rotation=30)
    plt.tight_layout()

    save_figure(fig, "attrition_percentage_by_department")

    typer.echo("\nAttrition Percentage by Department:")
    typer.echo(result.to_string(index=False))

    return result


def print_basic_stats(df: pd.DataFrame) -> None:
    """Print basic dataset statistics."""
    import typer
    typer.echo("\n=== Dataset Overview ===")
    typer.echo(f"Shape: {df.shape}")
    typer.echo(f"\nColumns ({len(df.columns)}): {list(df.columns)}")
    typer.echo(f"\nData Types:\n{df.dtypes.value_counts()}")
    typer.echo(f"\nMissing Values:\n{df.isnull().sum().sum()} total")
    typer.echo(f"\nTarget Distribution:\n{df['Attrition'].value_counts()}")
    typer.echo(f"\nTarget Distribution (%):\n{df['Attrition'].value_counts(normalize=True).mul(100).round(2)}")


if __name__ == "__main__":
    df = load_data()
    print_basic_stats(df)
    run_exploration(df)
