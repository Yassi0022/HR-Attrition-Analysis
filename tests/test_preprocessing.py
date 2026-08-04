"""Tests for the preprocessing module."""

import numpy as np
import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from attrition.config import (
    COLUMNS_TO_DROP,
    RANDOM_STATE,
    TARGET_COLUMN,
    TEST_SIZE,
)
from attrition.preprocessing import (
    create_preprocessing_pipeline,
    get_feature_columns,
    prepare_data,
    prepare_full_dataset,
)


@pytest.fixture
def sample_raw_data():
    """Create sample raw data matching the HR attrition schema."""
    np.random.seed(42)
    n = 100

    return pd.DataFrame(
        {
            "EmployeeNumber": range(1, n + 1),
            "EmployeeCount": [1] * n,
            "Over18": ["Y"] * n,
            "StandardHours": [80] * n,
            "Attrition": np.random.choice(["Yes", "No"], n, p=[0.2, 0.8]),
            "Age": np.random.randint(18, 65, n),
            "BusinessTravel": np.random.choice(
                ["Travel_Rarely", "Travel_Frequently", "Non-Travel"], n
            ),
            "Department": np.random.choice(["Sales", "Research", "HR"], n),
            "DistanceFromHome": np.random.randint(1, 30, n),
            "Education": np.random.randint(1, 5, n),
            "EducationField": np.random.choice(
                ["Life Sciences", "Medical", "Marketing", "Technical", "Other"], n
            ),
            "Gender": np.random.choice(["Male", "Female"], n),
            "JobRole": np.random.choice(
                ["Sales Rep", "Research Scientist", "Manager", "Developer"], n
            ),
            "MaritalStatus": np.random.choice(["Single", "Married", "Divorced"], n),
            "MonthlyIncome": np.random.randint(2000, 20000, n),
            "OverTime": np.random.choice(["Yes", "No"], n),
            "YearsAtCompany": np.random.randint(0, 20, n),
            # Add more numerical columns
            "DailyRate": np.random.randint(100, 1500, n),
            "HourlyRate": np.random.randint(30, 100, n),
            "MonthlyRate": np.random.randint(5000, 30000, n),
            "NumCompaniesWorked": np.random.randint(0, 10, n),
            "PercentSalaryHike": np.random.randint(5, 25, n),
            "PerformanceRating": np.random.randint(1, 5, n),
            "StockOptionLevel": np.random.randint(0, 3, n),
            "TotalWorkingYears": np.random.randint(0, 30, n),
            "TrainingTimesLastYear": np.random.randint(0, 6, n),
            "WorkLifeBalance": np.random.randint(1, 5, n),
            "YearsInCurrentRole": np.random.randint(0, 15, n),
            "YearsSinceLastPromotion": np.random.randint(0, 10, n),
            "YearsWithCurrManager": np.random.randint(0, 15, n),
        }
    )


def test_get_feature_columns(sample_raw_data):
    """Test feature column identification."""
    cat_cols, num_cols = get_feature_columns(sample_raw_data)

    # Should identify categorical columns
    assert "BusinessTravel" in cat_cols
    assert "Department" in cat_cols
    assert "Gender" in cat_cols

    # Should identify numerical columns
    assert "Age" in num_cols
    assert "MonthlyIncome" in num_cols
    assert "DistanceFromHome" in num_cols

    # Should NOT include dropped columns
    for col in COLUMNS_TO_DROP:
        assert col not in cat_cols
        assert col not in num_cols


def test_create_preprocessing_pipeline():
    """Test preprocessing pipeline creation."""
    cat_cols = ["BusinessTravel", "Department"]
    num_cols = ["Age", "MonthlyIncome"]

    pipeline = create_preprocessing_pipeline(cat_cols, num_cols)

    assert isinstance(pipeline, ColumnTransformer)

    # Check transformers (using 'transformers' attribute, not 'transformers_' in newer sklearn)
    transformer_names = [t[0] for t in pipeline.transformers]
    assert "cat" in transformer_names
    assert "num" in transformer_names

    # Check cat transformer is OneHotEncoder - access via transformers list (index 0 for 'cat', 1 for 'num')
    cat_transformer = pipeline.transformers[0][1]  # (name, transformer, columns)
    assert isinstance(cat_transformer, OneHotEncoder)
    assert cat_transformer.drop == "first"
    assert cat_transformer.handle_unknown == "ignore"

    # Check num transformer is StandardScaler
    num_transformer = pipeline.transformers[1][1]
    assert isinstance(num_transformer, StandardScaler)


def test_prepare_data_no_leakage(sample_raw_data):
    """Test that prepare_data avoids data leakage."""
    X_train, X_test, y_train, y_test, preprocessor = prepare_data(
        sample_raw_data,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )

    # Check shapes
    assert len(X_train) > 0
    assert len(X_test) > 0
    assert (
        len(X_train) + len(X_test)
        == len(sample_raw_data) - sample_raw_data[TARGET_COLUMN].isna().sum()
    )

    # Check target is binary
    assert set(y_train.unique()).issubset({0, 1})
    assert set(y_test.unique()).issubset({0, 1})

    # Check stratification roughly preserved
    train_yes_rate = (y_train == 1).mean()
    test_yes_rate = (y_test == 1).mean()
    overall_yes_rate = (sample_raw_data[TARGET_COLUMN] == "Yes").mean()

    # Should be roughly similar (within 5%)
    assert abs(train_yes_rate - overall_yes_rate) < 0.05
    assert abs(test_yes_rate - overall_yes_rate) < 0.05

    # Check preprocessor is fitted
    assert hasattr(preprocessor, "transformers_")

    # Check feature names
    assert len(X_train.columns) > 0
    assert list(X_train.columns) == list(X_test.columns)


def test_prepare_full_dataset(sample_raw_data):
    """Test full dataset preparation for feature importance."""
    X, y, preprocessor = prepare_full_dataset(sample_raw_data)

    assert len(X) == len(sample_raw_data)
    assert len(y) == len(sample_raw_data)
    assert set(y.unique()).issubset({0, 1})

    # Preprocessor should be fitted
    assert hasattr(preprocessor, "transformers_")


def test_target_mapping(sample_raw_data):
    """Test that target is correctly mapped to 0/1."""
    X_train, X_test, y_train, y_test, _ = prepare_data(sample_raw_data)

    # Original had "Yes"/"No", should be 1/0
    assert 1 in y_train.values  # Yes -> 1
    assert 0 in y_train.values  # No -> 0
    assert "Yes" not in y_train.values
    assert "No" not in y_train.values


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
