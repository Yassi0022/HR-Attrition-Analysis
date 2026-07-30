"""Configuration constants for the attrition analysis pipeline."""

from pathlib import Path

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent.parent

# Data paths
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_PATH = DATA_DIR / "WA_Fn-UseC_-HR-Employee-Attrition.csv"

# Output directories
IMG_DIR = PROJECT_ROOT / "img"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"

# Column names
TARGET_COLUMN = "Attrition"
ID_COLUMNS = ["EmployeeNumber", "EmployeeCount", "Over18", "StandardHours"]
COLUMNS_TO_DROP = ID_COLUMNS + [TARGET_COLUMN]

# Categorical columns that need encoding
CATEGORICAL_COLUMNS = [
    "BusinessTravel",
    "Department",
    "EducationField",
    "Gender",
    "JobRole",
    "MaritalStatus",
    "OverTime",
]

# Numerical columns for scaling
NUMERICAL_COLUMNS = [
    "Age",
    "DailyRate",
    "DistanceFromHome",
    "Education",
    "EnvironmentSatisfaction",
    "HourlyRate",
    "JobInvolvement",
    "JobLevel",
    "JobSatisfaction",
    "MonthlyIncome",
    "MonthlyRate",
    "NumCompaniesWorked",
    "PercentSalaryHike",
    "PerformanceRating",
    "RelationshipSatisfaction",
    "StockOptionLevel",
    "TotalWorkingYears",
    "TrainingTimesLastYear",
    "WorkLifeBalance",
    "YearsAtCompany",
    "YearsInCurrentRole",
    "YearsSinceLastPromotion",
    "YearsWithCurrManager",
]

# Model parameters
RANDOM_STATE = 42
TEST_SIZE = 0.2
LOGISTIC_REGRESSION_MAX_ITER = 10000
N_JOBS = -1

# Visualization settings
FIGURE_DPI = 150
FIGURE_FORMAT = "png"
DEFAULT_FIGSIZE = (10, 6)
LARGE_FIGSIZE = (14, 10)

# Ensure output directories exist
for directory in [IMG_DIR, MODELS_DIR, REPORTS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)
