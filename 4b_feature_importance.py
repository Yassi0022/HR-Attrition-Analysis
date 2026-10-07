import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Usa il file con le feature dai nomi reali
importance_df = pd.read_csv("models/feature_importance_full_dataset.csv")
importance_df = importance_df.sort_values(by="Importance", ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(data=importance_df.head(15), x="Importance", y="Feature")
plt.title("Top 15 Feature Importance - Random Forest")
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.tight_layout()

if not os.path.exists("img"):
    os.mkdir("img")

plt.savefig("img/feature_importance_random_forest.png", dpi=300)
plt.close()
print("Saved img/feature_importance_random_forest.png")
