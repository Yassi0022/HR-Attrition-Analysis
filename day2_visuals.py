import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Load dataset
df = pd.read_csv("WA_Fn-UseC_-HR-Employee-Attrition.csv")

# Ensure 'img' folder exists
if not os.path.exists("img"):
    os.makedirs("img")

# 1. Age distribution (histogram + KDE)
plt.figure(figsize=(8, 5))
sns.histplot(df["Age"], kde=True, bins=20)
plt.title("Age Distribution with KDE")
plt.xlabel("Age")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig("img/age_distribution.png")
plt.close()

# 2. Monthly Income boxplot
plt.figure(figsize=(8, 5))
sns.boxplot(x=df["MonthlyIncome"])
plt.title("Monthly Income Boxplot")
plt.xlabel("Monthly Income")
plt.tight_layout()
plt.savefig("img/monthly_income_boxplot.png")
plt.close()

# 3. Income by Attrition (boxplot)
plt.figure(figsize=(8, 5))
sns.boxplot(x="Attrition", y="MonthlyIncome", data=df)
plt.title("Monthly Income by Attrition")
plt.xlabel("Attrition")
plt.ylabel("Monthly Income")
plt.tight_layout()
plt.savefig("img/income_by_attrition.png")
plt.close()

print("✅ All visuals for Day 2 have been saved in 'img/' folder.")