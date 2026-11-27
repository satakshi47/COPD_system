"""Quick check for explanation.py and database.py WITHOUT running Streamlit.

Run from the project folder:   python try_explanation.py

It takes the first row of the test data, asks for an explanation (Gemini if
GEMINI_API_KEY is set in .env, otherwise the built-in template) and saves the
result to a separate test database (copd_test.db, ignored by git).
"""

import joblib
import pandas as pd

import database
import explanation

TEST_DB = "copd_test.db"

model = joblib.load("models/decision_tree.joblib")
names = pd.read_csv("data/processed/X_train.csv").columns.tolist()
row = pd.read_csv("data/processed/X_test.csv").iloc[[0]]

facts = explanation.get_decision_facts(model, row, names, defaulted_fields=[])
result = explanation.generate_explanation(facts)

print("Predicted class :", facts["predicted_class"])
print("Source          :", result["source"])
print("Problem         :", result["error"] or "none")
print("\nExplanation:\n", result["text"])

database.init_db(TEST_DB)
record_id = database.save_prediction(
    model_name="Decision Tree",
    predicted_class=facts["predicted_class"],
    inputs=row.iloc[0].to_dict(),
    defaulted_fields=[],
    decision_path=facts["decision_path"],
    explanation=result["text"],
    explanation_source=result["source"],
    prompt_version=explanation.PROMPT_VERSION,
    db_path=TEST_DB,
)
saved = database.get_recent(1, TEST_DB)[0]
print(f"\nDatabase OK: saved record #{record_id}, read back class = {saved['predicted_class']}")