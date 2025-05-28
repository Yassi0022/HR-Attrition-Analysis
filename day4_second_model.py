import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder



df = pd.read_csv("WA_Fn-UseC_-HR-Employee-Attrition.csv")

# Encoding variabili categoriche
df_encoded = df.copy()
for column in df_encoded.select_dtypes(include="object").columns:
    df_encoded[column] = LabelEncoder().fit_transform(df_encoded[column])

# Separazione X e y
X = df_encoded.drop(columns=["Attrition"])
y = df_encoded["Attrition"]

# Split train-test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

importances = model.feature_importances_
features = X.columns

# Crea DataFrame ordinato
importance_df = pd.DataFrame({
    "Feature": features,
    "Importance": importances
}).sort_values(by="Importance", ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(data=importance_df.head(15), x="Importance", y="Feature")
plt.title("Top 15 Features Influencing Attrition")
plt.tight_layout()

# Crea cartella immagini se serve
import os
if not os.path.exists("img"):
    os.mkdir("img")

plt.savefig("img/feature_importance.png")
plt.show()

