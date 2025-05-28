import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("WA_Fn-UseC_-HR-Employee-Attrition.csv")
df_numeric = df.select_dtypes(include="number")

corr_matrix = df_numeric.corr()

plt.figure(figsize=(14,10))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Correlation Matrix – Numerical Variables")
plt.tight_layout()
plt.savefig("img/correlations.png")
plt.close()

