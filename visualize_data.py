import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Load dataset
df = pd.read_csv("dataset/adult.csv")

# Create output folder
os.makedirs("visualizations", exist_ok=True)

# Remove duplicate rows
df = df.drop_duplicates()

# 1. Income Distribution
plt.figure(figsize=(8, 5))
sns.countplot(data=df, x="income")
plt.title("Income Distribution")
plt.xlabel("Income")
plt.ylabel("Number of People")
plt.tight_layout()
plt.savefig("visualizations/income_distribution.png")
plt.close()

# 2. Age Distribution
plt.figure(figsize=(8, 5))
sns.histplot(data=df, x="age", bins=30, kde=True)
plt.title("Age Distribution")
plt.xlabel("Age")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("visualizations/age_distribution.png")
plt.close()

# 3. Education vs Income
plt.figure(figsize=(12, 6))
sns.countplot(
    data=df,
    x="education",
    hue="income"
)
plt.title("Education vs Income")
plt.xlabel("Education")
plt.ylabel("Count")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("visualizations/education_vs_income.png")
plt.close()

# 4. Hours per Week vs Income
plt.figure(figsize=(8, 5))
sns.boxplot(
    data=df,
    x="income",
    y="hours-per-week"
)
plt.title("Working Hours vs Income")
plt.xlabel("Income")
plt.ylabel("Hours per Week")
plt.tight_layout()
plt.savefig("visualizations/hours_vs_income.png")
plt.close()

# 5. Income by Sex
plt.figure(figsize=(8, 5))
sns.countplot(
    data=df,
    x="sex",
    hue="income"
)
plt.title("Income Distribution by Sex")
plt.xlabel("Sex")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("visualizations/income_by_sex.png")
plt.close()

print("=" * 50)
print("VISUALIZATION COMPLETED")
print("=" * 50)
print("Charts saved in: visualizations/")

