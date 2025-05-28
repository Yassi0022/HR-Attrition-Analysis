import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

df = pd.read_csv("WA_Fn-UseC_-HR-Employee-Attrition.csv")

#print(df.info())
#print(df.head())
#print(df.describe())

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

if not os.path.exists("img"):
    os.mkdir("img")

plt.figure(figsize=(8, 5))
sns.barplot(data=result, x="Department", y="Attrition %")

plt.title("Attrition Rate by Department")
plt.ylabel("Attrition %")
plt.xlabel("Department")
plt.xticks(rotation=30)
plt.tight_layout()


plt.savefig("img/attriction_percentage_by_department.png")
plt.show()
plt.close()


print("\n Attrition percentage by department:")
print(result)
