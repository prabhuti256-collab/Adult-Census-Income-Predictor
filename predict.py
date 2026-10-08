import pandas as pd
import joblib

# Load trained model
model = joblib.load("models/income_model.pkl")

# Sample person
person = pd.DataFrame([{
    "age": 35,
    "workclass": "Private",
    "fnlwgt": 180000,
    "education": "Bachelors",
    "education-num": 13,
    "marital-status": "Married-civ-spouse",
    "occupation": "Exec-managerial",
    "relationship": "Husband",
    "race": "White",
    "sex": "Male",
    "capital-gain": 0,
    "capital-loss": 0,
    "hours-per-week": 45,
    "native-country": "United-States"
}])

# Prediction
prediction = model.predict(person)[0]

# Probability
probability = model.predict_proba(person)[0]

print("=" * 60)
print("        ADULT CENSUS INCOME PREDICTION")
print("=" * 60)

if prediction == 1:
    print("\nPredicted Income: >50K")
else:
    print("\nPredicted Income: <=50K")

print(f"Probability of <=50K: {probability[0] * 100:.2f}%")
print(f"Probability of >50K : {probability[1] * 100:.2f}%")

print("=" * 60)

