"""Data preprocessing pipeline for attrition analysis."""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from attrition.config import (
    CATEGORICAL_COLUMNS,
    COLUMNS_TO_DROP,
    NUMERICAL_COLUMNS,
    RANDOM_STATE,
    TARGET_COLUMN,
    TEST_SIZE,
)
from attrition.utils import load_data


def get_feature_columns(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Identify categorical and numerical columns in the dataset."""
    # Drop target and ID columns
    feature_df = df.drop(columns=[c for c in COLUMNS_TO_DROP if c in df.columns])

    # Identify column types
    cat_cols = [c for c in CATEGORICAL_COLUMNS if c in feature_df.columns]
    num_cols = [c for c in NUMERICAL_COLUMNS if c in feature_df.columns]

    # Add any other categorical columns not in our list
    other_cat = feature_df.select_dtypes(include=["object", "category"]).columns
    for col in other_cat:
        if col not in cat_cols:
            cat_cols.append(col)

    # Add any other numerical columns not in our list
    other_num = feature_df.select_dtypes(include=[np.number]).columns
    for col in other_num:
        if col not in num_cols and col not in cat_cols:
            num_cols.append(col)

    return cat_cols, num_cols


def create_preprocessing_pipeline(
    categorical_cols: list[str],
    numerical_cols: list[str],
) -> ColumnTransformer:
    """
    Create a preprocessing pipeline that avoids data leakage.

    Key fix: StandardScaler is fitted ONLY on training data, not on the full dataset.
    """
    return ColumnTransformer(
        transformers=[
            (
                "cat",
                OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore"),
                categorical_cols,
            ),
            (
                "num",
                StandardScaler(),
                numerical_cols,
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def prepare_data(
    df: pd.DataFrame | None = None,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, ColumnTransformer]:
    """
    Prepare train/test split with proper preprocessing.

    Returns:
        X_train, X_test, y_train, y_test, fitted_preprocessor
    """
    if df is None:
        df = load_data()

    # Map target
    df = df.copy()
    df[TARGET_COLUMN] = df[TARGET_COLUMN].map({"Yes": 1, "No": 0})

    # Separate features and target
    X = df.drop(columns=[c for c in COLUMNS_TO_DROP if c in df.columns])
    y = df[TARGET_COLUMN]

    # Get column types
    cat_cols, num_cols = get_feature_columns(df)

    # Split FIRST, then preprocess (avoids data leakage)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Create and fit preprocessor on training data only
    preprocessor = create_preprocessing_pipeline(cat_cols, num_cols)
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    # Get feature names after transformation
    feature_names = preprocessor.get_feature_names_out()

    return (
        pd.DataFrame(X_train_processed, columns=feature_names, index=X_train.index),
        pd.DataFrame(X_test_processed, columns=feature_names, index=X_test.index),
        y_train,
        y_test,
        preprocessor,
    )


def prepare_full_dataset(
    df: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.Series, ColumnTransformer]:
    """Prepare the full dataset for feature importance analysis (no train/test split)."""
    if df is None:
        df = load_data()

    df = df.copy()
    df[TARGET_COLUMN] = df[TARGET_COLUMN].map({"Yes": 1, "No": 0})

    X = df.drop(columns=[c for c in COLUMNS_TO_DROP if c in df.columns])
    y = df[TARGET_COLUMN]

    cat_cols, num_cols = get_feature_columns(df)
    preprocessor = create_preprocessing_pipeline(cat_cols, num_cols)
    X_processed = preprocessor.fit_transform(X)
    feature_names = preprocessor.get_feature_names_out()

    return pd.DataFrame(X_processed, columns=feature_names, index=X.index), y, preprocessor
