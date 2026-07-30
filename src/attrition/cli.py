"""Command-line interface for the attrition analysis pipeline."""

from pathlib import Path

import pandas as pd
import typer

from attrition.config import RAW_DATA_PATH
from attrition.correlations import run_correlation_analysis
from attrition.exploration import run_exploration
from attrition.model import (
    run_feature_importance_analysis,
    run_training_pipeline,
)
from attrition.utils import load_data
from attrition.visuals import run_all_visualizations

app = typer.Typer(
    name="attrition",
    help="HR Attrition Analysis Pipeline - IBM HR Analytics Employee Attrition Dataset",
    add_completion=False,
    no_args_is_help=True,
)


@app.callback()
def main(
    data_path: Path | None = typer.Option(
        RAW_DATA_PATH,
        "--data-path",
        "-d",
        help="Path to the HR Attrition CSV dataset",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
    ),
):
    """
    HR Attrition Analysis Pipeline

    Analyzes the IBM HR Analytics Employee Attrition dataset to identify
    factors contributing to employee attrition and build predictive models.
    """
    pass


@app.command()
def explore(
    data_path: Path | None = typer.Option(
        RAW_DATA_PATH,
        "--data-path",
        "-d",
        help="Path to the HR Attrition CSV dataset",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
    ),
):
    """Run exploratory data analysis and generate attrition by department visualization."""
    typer.echo("Running exploratory data analysis...")
    df = load_data(data_path)
    run_exploration(df)
    typer.echo("[OK] Exploration complete!")


@app.command()
def visualize(
    data_path: Path | None = typer.Option(
        RAW_DATA_PATH,
        "--data-path",
        "-d",
        help="Path to the HR Attrition CSV dataset",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
    ),
):
    """Generate all visualizations (age, income, attrition by categorical features)."""
    typer.echo("Generating visualizations...")
    df = load_data(data_path)
    run_all_visualizations(df)
    typer.echo("[OK] Visualizations complete!")


@app.command()
def correlate(
    data_path: Path | None = typer.Option(
        RAW_DATA_PATH,
        "--data-path",
        "-d",
        help="Path to the HR Attrition CSV dataset",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
    ),
    threshold: float = typer.Option(
        0.7,
        "--threshold",
        "-t",
        help="Correlation threshold for strong correlations",
    ),
):
    """Run correlation analysis and generate heatmap."""
    typer.echo("Running correlation analysis...")
    df = load_data(data_path)
    run_correlation_analysis(df)
    typer.echo("[OK] Correlation analysis complete!")


@app.command()
def train(
    data_path: Path | None = typer.Option(
        RAW_DATA_PATH,
        "--data-path",
        "-d",
        help="Path to the HR Attrition CSV dataset",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
    ),
    models: list[str] = typer.Option(
        ["logistic_regression", "random_forest"],
        "--model",
        "-m",
        help="Models to train (can be specified multiple times)",
    ),
):
    """Train predictive models (Logistic Regression and/or Random Forest)."""
    typer.echo(f"Training models: {', '.join(models)}")
    df = load_data(data_path)
    run_training_pipeline(df, models_to_train=models)
    typer.echo("[OK] Training complete!")


@app.command()
def feature_importance(
    data_path: Path | None = typer.Option(
        RAW_DATA_PATH,
        "--data-path",
        "-d",
        help="Path to the HR Attrition CSV dataset",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
    ),
):
    """Run feature importance analysis using Random Forest on full dataset."""
    typer.echo("Running feature importance analysis...")
    df = load_data(data_path)
    run_feature_importance_analysis(df)
    typer.echo("[OK] Feature importance analysis complete!")


@app.command()
def all(
    data_path: Path | None = typer.Option(
        RAW_DATA_PATH,
        "--data-path",
        "-d",
        help="Path to the HR Attrition CSV dataset",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
    ),
):
    """Run the complete analysis pipeline: explore, visualize, correlate, train."""
    typer.echo("Running complete analysis pipeline...")

    typer.echo("\n[1/4] Exploratory Data Analysis...")
    df = load_data(data_path)
    run_exploration(df)

    typer.echo("\n[2/4] Visualizations...")
    run_all_visualizations(df)

    typer.echo("\n[3/4] Correlation Analysis...")
    run_correlation_analysis(df)

    typer.echo("\n[4/4] Model Training...")
    run_training_pipeline(df)

    typer.echo("\n[OK] Complete pipeline finished!")


@app.command()
def predict(
    model_name: str = typer.Argument(
        ...,
        help="Model to use for prediction (logistic_regression or random_forest)",
    ),
    input_file: Path = typer.Argument(
        ...,
        help="CSV file with employee data to predict",
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
    ),
    output_file: Path | None = typer.Option(
        None,
        "--output",
        "-o",
        help="Output CSV file for predictions",
    ),
):
    """Make predictions on new employee data using a trained model."""
    from attrition.model import load_model

    typer.echo(f"Loading model: {model_name}...")
    model, preprocessor, metrics = load_model(model_name)

    typer.echo(f"Loading data from: {input_file}...")
    _ = pd.read_csv(input_file)

    # Prepare data (need to match training preprocessing)
    # Note: In production, you'd save the preprocessor and use it directly
    # This is a simplified version for demo purposes
    typer.echo("Making predictions...")
    # This would need the full preprocessing pipeline to work correctly
    # For now, we just show the structure
    typer.echo("Note: Full prediction requires the saved preprocessor pipeline.")
    typer.echo("This is a placeholder for production deployment.")


if __name__ == "__main__":
    app()
