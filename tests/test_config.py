"""Tests for configuration module."""


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


def test_project_root_exists():
    """Test that PROJECT_ROOT points to the correct directory."""
    assert PROJECT_ROOT.exists()
    assert PROJECT_ROOT.name == "attrition"


def test_data_paths():
    """Test that data paths are correctly configured."""
    assert DATA_DIR == PROJECT_ROOT / "data"
    assert RAW_DATA_PATH == DATA_DIR / "WA_Fn-UseC_-HR-Employee-Attrition.csv"
    assert RAW_DATA_PATH.exists()


def test_output_directories():
    """Test that output directories are correctly configured."""
    assert IMG_DIR == PROJECT_ROOT / "img"
    assert MODELS_DIR == PROJECT_ROOT / "models"
    assert REPORTS_DIR == PROJECT_ROOT / "reports"
    assert IMG_DIR.exists()
    assert MODELS_DIR.exists()
    assert REPORTS_DIR.exists()


def test_column_configs():
    """Test column configuration constants."""
    assert TARGET_COLUMN == "Attrition"
    assert "EmployeeNumber" in COLUMNS_TO_DROP
    assert "EmployeeCount" in COLUMNS_TO_DROP
    assert "Over18" in COLUMNS_TO_DROP
    assert "StandardHours" in COLUMNS_TO_DROP
    assert TARGET_COLUMN in COLUMNS_TO_DROP


def test_categorical_columns():
    """Test categorical columns list."""
    expected_categorical = [
        "BusinessTravel",
        "Department",
        "EducationField",
        "Gender",
        "JobRole",
        "MaritalStatus",
        "OverTime",
    ]
    assert expected_categorical == CATEGORICAL_COLUMNS


def test_numerical_columns():
    """Test numerical columns list."""
    assert isinstance(NUMERICAL_COLUMNS, list)
    assert len(NUMERICAL_COLUMNS) > 20
    assert "Age" in NUMERICAL_COLUMNS
    assert "MonthlyIncome" in NUMERICAL_COLUMNS
    assert "YearsAtCompany" in NUMERICAL_COLUMNS


def test_model_params():
    """Test model parameter constants."""
    assert RANDOM_STATE == 42
    assert TEST_SIZE == 0.2


def test_visualization_params():
    """Test visualization parameter constants."""
    assert FIGURE_DPI == 150
    assert FIGURE_FORMAT == "png"
    assert DEFAULT_FIGSIZE == (10, 6)
    assert LARGE_FIGSIZE == (14, 10)
