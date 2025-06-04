import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix


df = pd.read_csv("WA_Fn-UseC_-HR-Employee-Attrition.csv")
df = df.drop(columns=["EmployeeNumber", "EmployeeCount", "Over18", "StandardHours"])
df["Attrition"] = df["Attrition"].map({"Yes":1, "No":0})

X = df.drop(columns=["Attrition"])
Y = df["Attrition"]
X = pd.get_dummies(X)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Use the scaled features for the logistic regression model
X_train, X_test, Y_train, Y_test = train_test_split(
    X_scaled, Y, test_size=0.2, random_state=42
)
#print("Training set size (scaled):", X_train.shape[0])
#print("Test set size (scaled):", X_test.shape[0])

model = LogisticRegression(max_iter=10000, class_weight="balanced")
model.fit(X_train, Y_train)

Y_pred = model.predict(X_test)

accuracy = accuracy_score(Y_test, Y_pred)
#print("Model accuracy:", round(accuracy * 100, 2), "%")
#print(Y.value_counts(normalize=True))

print("\nConfusion Matrix:")
print(confusion_matrix(Y_test, Y_pred))

print("\nClassification Report:")
print(classification_report(Y_test, Y_pred))
