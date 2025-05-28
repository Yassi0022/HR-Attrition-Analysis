# HR Attrition Analysis

This project analyzes employee attrition in a company using the IBM HR Analytics dataset.

##  Goals
- Identify which departments experience the highest attrition rates
- Visualize attrition percentage by department
- Provide actionable insights for HR strategy

##  Technologies Used
- Python
- Pandas
- Matplotlib
- Seaborn

##  Key Insights
- The **Sales department** has the highest attrition rate (~20%)
- **Human Resources** also shows high turnover compared to other areas
- R&D has the **lowest attrition**, suggesting higher stability

##  Dataset
- [IBM HR Analytics Dataset](https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset)

---

## 2 – Visual Analysis by Department

We visualized attrition rates by department to identify where turnover is most critical.

### Key Findings:
- **Sales** has the highest attrition rate (~20%), indicating a need for targeted HR interventions.
- **Human Resources** also shows elevated attrition compared to R&D.
- **R&D** appears the most stable department overall.

---

## 3 – Correlation Insights

- Strong positive correlation between `JobLevel` and `MonthlyIncome` (~0.95): higher-level employees earn more.
- `TotalWorkingYears` strongly correlates with `Age` (~0.78): older employees have more experience.
- `EnvironmentSatisfaction`, `JobSatisfaction`, and `WorkLifeBalance` show low correlations with other features – suggesting independent psychological dimensions.

---

## 4 – Machine Learning Model

- **Model used:** Logistic Regression
- **Accuracy achieved:** ~88%
- **Limitation:** High class imbalance; only ~16% of employees actually leave
- **Adjusted model:** Used `class_weight="balanced"` to improve fairness
- **Outcome:** Precision and recall on attrition improved, even if accuracy dropped



## 4b – Feature Importance Insights

Using a Random Forest model, we identified the most influential variables in employee attrition.

### Top Influential Features:
1. **MonthlyIncome** – Lower salaries correlate with higher attrition
2. **OverTime** – Frequent overtime increases attrition risk
3. **Age** – Younger employees are more likely to leave
4. **DistanceFromHome** – Long commute contributes to dissatisfaction
5. **YearsAtCompany** – New employees are more likely to quit

### HR Recommendations:
- Limit excessive overtime to reduce burnout
- Offer competitive early-career salary packages
- Improve onboarding and mentorship for new hires
- Consider hybrid work or transportation support for distant workers


## Next Steps

- Apply more advanced models like Random Forest and XGBoost
- Use cross-validation and SMOTE to handle imbalance
- Build an interactive dashboard to present insights