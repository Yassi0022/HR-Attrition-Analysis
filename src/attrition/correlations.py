"""Correlation analysis module for attrition analysis."""

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import typer

from attrition.config import IMG_DIR
from attrition.utils import ensure_dir, load_data, save_figure


def compute_correlation_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Compute correlation matrix for numerical columns."""
    numeric_df = df.select_dtypes(include="number")
    return numeric_df.corr()


def plot_correlation_heatmap(
    corr_matrix: pd.DataFrame,
    title: str = "Correlation Matrix – Numerical Variables",
    figsize: tuple = (14, 10),
    annot: bool = True,
    fmt: str = ".2f",
    cmap: str = "coolwarm",
) -> None:
    """Plot and save correlation heatmap."""
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        corr_matrix,
        annot=annot,
        cmap=cmap,
        fmt=fmt,
        linewidths=0.5,
        ax=ax,
        cbar_kws={"shrink": 0.8},
    )
    ax.set_title(title, fontsize=14, fontweight="bold")
    plt.tight_layout()
    save_figure(fig, "correlations")


def find_strong_correlations(
    corr_matrix: pd.DataFrame,
    threshold: float = 0.7,
) -> pd.DataFrame:
    """Find pairs of features with correlation above threshold."""
    pairs = []
    cols = corr_matrix.columns
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            corr_val = corr_matrix.iloc[i, j]
            if abs(corr_val) >= threshold:
                pairs.append(
                    {
                        "Feature_1": cols[i],
                        "Feature_2": cols[j],
                        "Correlation": corr_val,
                    }
                )
    if not pairs:
        return pd.DataFrame(columns=["Feature_1", "Feature_2", "Correlation"])
    return pd.DataFrame(pairs).sort_values("Correlation", key=abs, ascending=False)


def run_correlation_analysis(df: pd.DataFrame = None) -> pd.DataFrame:
    """Run full correlation analysis and generate visualizations."""
    if df is None:
        df = load_data()

    ensure_dir(IMG_DIR)

    typer.echo("Computing correlation matrix...")
    corr_matrix = compute_correlation_matrix(df)

    typer.echo("Generating heatmap...")
    plot_correlation_heatmap(corr_matrix)

    typer.echo("Finding strong correlations...")
    strong_corr = find_strong_correlations(corr_matrix, threshold=0.7)

    if not strong_corr.empty:
        typer.echo("\nStrong Correlations (|r| >= 0.7):")
        typer.echo(strong_corr.to_string(index=False))
    else:
        typer.echo("\nNo strong correlations found (|r| >= 0.7)")

    return corr_matrix


if __name__ == "__main__":
    df = load_data()
    run_correlation_analysis(df)
