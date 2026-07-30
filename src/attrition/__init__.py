"""
HR Attrition Analysis Package

End-to-end ML pipeline for predicting employee attrition using the
IBM HR Analytics Employee Attrition dataset.
"""

from attrition.config import (
    CATEGORICAL_COLUMNS,
    COLUMNS_TO_DROP,
    DATA_DIR,
    DEFAULT_FIGSIZE,
    FIGURE_DPI,
    FIGURE_FORMAT,
    IMG_DIR,
    LARGE_FIGSIZE,
    MODELS_DIR,
    NUMERICAL_COLUMNS,
    PROJECT_ROOT,
    RANDOM_STATE,
    RAW_DATA_PATH,
    REPORTS_DIR,
    TARGET_COLUMN,
    TEST_SIZE,
)
from attrition.correlations import (
    compute_correlation_matrix,
    find_strong_correlations,
    plot_correlation_heatmap,
    run_correlation_analysis,
)
from attrition.exploration import (
    print_basic_stats,
    run_exploration,
)
from attrition.model import (
    evaluate_model,
    get_feature_importance,
    plot_confusion_matrix,
    plot_feature_importance,
    plot_logistic_coefficients,
    plot_roc_curve,
    run_feature_importance_analysis,
    run_training_pipeline,
    train_logistic_regression,
    train_random_forest,
)
from attrition.model import (
    load_model as load_trained_model,
)
from attrition.model import (
    save_model as save_trained_model,
)
from attrition.preprocessing import (
    create_preprocessing_pipeline,
    get_feature_columns,
    prepare_data,
    prepare_full_dataset,
)
from attrition.utils import (
    ensure_dir,
    load_data,
    load_model,
    save_dataframe,
    save_figure,
    save_model,
)
from attrition.visuals import (
    plot_age_distribution,
    plot_attrition_by_categorical,
    plot_income_by_attrition,
    plot_monthly_income_boxplot,
    run_all_visualizations,
)

__version__ = "0.1.0"
__author__ = "Yassi"
__all__ = [
    # Config
    "PROJECT_ROOT",
    "DATA_DIR",
    "RAW_DATA_PATH",
    "IMG_DIR",
    "MODELS_DIR",
    "REPORTS_DIR",
    "TARGET_COLUMN",
    "COLUMNS_TO_DROP",
    "CATEGORICAL_COLUMNS",
    "NUMERICAL_COLUMNS",
    "RANDOM_STATE",
    "TEST_SIZE",
    "FIGURE_DPI",
    "FIGURE_FORMAT",
    "DEFAULT_FIGSIZE",
    "LARGE_FIGSIZE",
    # Utils
    "ensure_dir",
    "save_figure",
    "load_data",
    "save_model",
    "load_model",
    "save_dataframe",
    # Preprocessing
    "get_feature_columns",
    "create_preprocessing_pipeline",
    "prepare_data",
    "prepare_full_dataset",
    # Exploration
    "run_exploration",
    "print_basic_stats",
    # Visuals
    "plot_age_distribution",
    "plot_monthly_income_boxplot",
    "plot_income_by_attrition",
    "plot_attrition_by_categorical",
    "run_all_visualizations",
    # Correlations
    "compute_correlation_matrix",
    "plot_correlation_heatmap",
    "find_strong_correlations",
    "run_correlation_analysis",
    # Model
    "train_logistic_regression",
    "train_random_forest",
    "evaluate_model",
    "plot_confusion_matrix",
    "plot_roc_curve",
    "get_feature_importance",
    "plot_feature_importance",
    "plot_logistic_coefficients",
    "save_trained_model",
    "load_trained_model",
    "run_training_pipeline",
    "run_feature_importance_analysis",
]
