import pandas as pd

# Load dataset
df = pd.read_csv("dataset/adult.csv")

print("=" * 60)
print("ADULT CENSUS INCOME DATASET")
print("=" * 60)

# Dataset shape
print("\nDataset Shape:")
print(df.shape)

# First 5 rows
print("\nFirst 5 Rows:")
print(df.head())

# Data types
print("\nData Types:")
print(df.dtypes)

# Missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Duplicate rows
print("\nDuplicate Rows:")
print(df.duplicated().sum())

# Income distribution
print("\nIncome Distribution:")
print(df["income"].value_counts())

# Income percentage
print("\nIncome Percentage:")
print(df["income"].value_counts(normalize=True) * 100)

# Numerical statistics
print("\nNumerical Statistics:")
print(df.describe())

print("\n" + "=" * 60)
print("DATA EXPLORATION COMPLETED")
print("=" * 60)
