import os
import joblib

import pandas as pd

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from sklearn.metrics import accuracy_score, f1_score, classification_report

# 1. Load processed data
X_train = pd.read_csv("data/processed/X_train.csv")
X_test = pd.read_csv("data/processed/X_test.csv")

y_train = pd.read_csv("data/processed/y_train.csv").squeeze()
y_test = pd.read_csv("data/processed/y_test.csv").squeeze()

# 2. Define models
models = {
    "Logistic Regression": make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000, class_weight="balanced")
    ),

    "Decision Tree": DecisionTreeClassifier(
        random_state=42,
        class_weight="balanced"
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        class_weight="balanced"
    ),

    "SVM": make_pipeline(
        StandardScaler(),
        SVC(class_weight="balanced")
    )
}

# 3. Train and evaluate each model
results = []

for name, model in models.items():
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    macro_f1 = f1_score(y_test, predictions, average="macro")

    results.append({
        "Model": name,
        "Accuracy": round(accuracy, 3),
        "Macro F1": round(macro_f1, 3)
    })

    print("\n", name)
    print(classification_report(y_test, predictions, zero_division=0))

# 4. Compare model results
results_df = pd.DataFrame(results)
print("\nModel Comparison:")
print(results_df)

# 5. Save trained models
os.makedirs("models", exist_ok=True)

for name, model in models.items():
    filename = name.lower().replace(" ", "_") + ".joblib"
    joblib.dump(model, f"models/{filename}")

print("\nAll models saved in the models folder.")