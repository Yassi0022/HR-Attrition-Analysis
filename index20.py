import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_csv("WA_Fn-UseC_-HR-Employee-Attrition.csv")
#print(df.head())

#print("\nDistribuzione abbandoni:")
#print(df["Attrition"].value_counts())

#print("\nEtà media per abbandoni:")
#print(df.groupby("Attrition")["Age"].mean())

#print("\nTabella abbandoni per dipartimento:")
attrition_ct = pd.crosstab(df["Department"], df["Attrition"])
attrition_percentage = attrition_ct.apply(lambda row: row["Yes"] / row.sum() * 100, axis=1)

result = pd.DataFrame({
    "Department" : attrition_percentage.index,
    "Attrition %": attrition_percentage.values
})
plt.figure(figsize=(8, 5))
sns.barplot(data=result, x="Department", y="Attrition %")

plt.title("Attrition Rate by Department")
plt.ylabel("Attrition %")
plt.xlabel("Department")
plt.xticks(rotation=30)
plt.tight_layout()
plt.show()



print("\n Attrition percentage by department:")
print(result)
