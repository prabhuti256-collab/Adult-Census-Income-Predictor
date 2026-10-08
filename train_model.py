import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)


# ============================================================
# 1. LOAD DATASET
# ============================================================

df = pd.read_csv("dataset/adult.csv")

print("=" * 65)
print("        ADULT CENSUS INCOME PREDICTOR")
print("=" * 65)

print("\nOriginal dataset shape:", df.shape)


# ============================================================
# 2. REMOVE DUPLICATES
# ============================================================

df = df.drop_duplicates()

print("After removing duplicates:", df.shape)


# ============================================================
# 3. FEATURES AND TARGET
# ============================================================

X = df.drop("income", axis=1)

y = df["income"].map({
    "<=50K": 0,
    ">50K": 1
})


# ============================================================
# 4. IDENTIFY FEATURES
# ============================================================

categorical_features = X.select_dtypes(
    include=["object", "string", "category"]
).columns.tolist()

numerical_features = X.select_dtypes(
    exclude=["object", "string", "category"]
).columns.tolist()


print("\nNumerical Features:")
print(numerical_features)

print("\nCategorical Features:")
print(categorical_features)


# ============================================================
# 5. PREPROCESSING
# ============================================================

preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            "passthrough",
            numerical_features
        ),
        (
            "cat",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False
            ),
            categorical_features
        )
    ]
)


# ============================================================
# 6. TRAIN TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))


# ============================================================
# 7. DEFINE MODELS
# ============================================================

models = {

    "Logistic Regression": LogisticRegression(
        max_iter=1000,
        random_state=42
    ),

    "Decision Tree": DecisionTreeClassifier(
        max_depth=12,
        random_state=42
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        random_state=42,
        n_jobs=-1
    )
}


# ============================================================
# 8. TRAIN AND EVALUATE
# ============================================================

results = {}

best_model = None
best_model_name = None
best_f1 = 0


for name, model in models.items():

    print("\n" + "=" * 65)
    print(f"TRAINING: {name}")
    print("=" * 65)

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    # Train
    pipeline.fit(X_train, y_train)

    # Predictions
    y_pred = pipeline.predict(X_test)

    # Probabilities
    y_prob = pipeline.predict_proba(X_test)[:, 1]

    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)

    cm = confusion_matrix(y_test, y_pred)

    results[name] = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc
    }

    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")
    print(f"ROC-AUC  : {roc_auc:.4f}")

    print("\nConfusion Matrix:")
    print(cm)

    # Select best model using F1-score
    if f1 > best_f1:
        best_f1 = f1
        best_model = pipeline
        best_model_name = name


# ============================================================
# 9. MODEL COMPARISON
# ============================================================

print("\n\n")
print("=" * 65)
print("                 MODEL COMPARISON")
print("=" * 65)

for name, metrics in results.items():

    print(f"\n{name}")

    print(f"  Accuracy : {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall   : {metrics['recall']:.4f}")
    print(f"  F1 Score : {metrics['f1']:.4f}")
    print(f"  ROC-AUC  : {metrics['roc_auc']:.4f}")


# ============================================================
# 10. SAVE BEST MODEL
# ============================================================

print("\n" + "=" * 65)
print("                    BEST MODEL")
print("=" * 65)

print("Best Model:", best_model_name)
print(f"Best F1 Score: {best_f1:.4f}")


joblib.dump(
    best_model,
    "models/income_model.pkl"
)

print("\nBest model saved successfully!")
print("Location: models/income_model.pkl")

print("=" * 65)
