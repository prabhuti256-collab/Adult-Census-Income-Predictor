import pandas as pd

columns = [
    "age",
    "workclass",
    "fnlwgt",
    "education",
    "education-num",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "capital-gain",
    "capital-loss",
    "hours-per-week",
    "native-country",
    "income"
]

df = pd.read_csv(
    "dataset/adult.data",
    names=columns,
    skipinitialspace=True
)

# Remove rows with missing values
df = df.replace("?", pd.NA)
df = df.dropna()

# Remove trailing period from income values if present
df["income"] = df["income"].str.replace(".", "", regex=False)

# Save as CSV
df.to_csv("dataset/adult.csv", index=False)

print("Dataset created successfully!")
print("Shape:", df.shape)
print("\nColumns:")
print(df.columns.tolist())
print("\nIncome distribution:")
print(df["income"].value_counts())
