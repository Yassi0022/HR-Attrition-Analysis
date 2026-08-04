🎉 Excited to share my completed HR Attrition Analysis project — now live on GitHub!

🔗 Repository: https://github.com/Yassi0022/HR-Attrition-Analysis

This project analyzes the IBM HR Analytics Employee Attrition dataset to identify key factors driving employee turnover and build predictive models for retention strategies.

📊 **What's inside:**
- **Exploratory Data Analysis** — Age, income, attrition by department/role/overtime
- **Visual Analytics** — 12+ publication-ready charts (boxplots, KDEs, attrition heatmaps)
- **Correlation Analysis** — Multicollinearity detection with threshold-based filtering
- **Predictive Modeling** — Logistic Regression & Random Forest with proper preprocessing pipelines (no data leakage!)
- **Feature Importance** — Business-interpretable drivers of attrition
- **Production-ready CLI** — Run any analysis step with `attrition explore`, `attrition train`, etc.

🏗️ **Engineering highlights:**
- Modular package architecture (`src/attrition/`) — clean separation of concerns
- Zero data leakage — `ColumnTransformer` + `StandardScaler` fit on training data only
- Proper categorical encoding — `OneHotEncoder(drop='first', handle_unknown='ignore')`
- Comprehensive test suite (43 tests, 100% pass) with pytest
- CI/CD pipeline (GitHub Actions: ruff, mypy, pytest, build)
- Professional docs with architecture diagram, recruiter-focused features table
- Pinned dependencies, pyproject.toml packaging, Typer CLI

🛠️ **Stack:** Python, pandas, scikit-learn, matplotlib, seaborn, Typer, pytest, GitHub Actions

This project demonstrates end-to-end ML engineering practices — from reproducible preprocessing to deployable pipelines — exactly what hiring managers look for in a data science portfolio.

#DataScience #MachineLearning #HRAnalytics #EmployeeAttrition #Python #PortfolioProject #OpenSource #GitHub #JobSearch