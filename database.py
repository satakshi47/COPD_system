"""SQLite storage for COPD predictions and their explanations.

Uses only Python's built-in sqlite3. The database file is created next to this
file on first use and is already ignored by .gitignore (*.db).

No names, IDs or contact details are stored: only the model inputs and outputs.
"""

import json
import sqlite3
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path

DB_PATH = Path(__file__).parent / "copd_predictions.db"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS predictions (
    id                    INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at            TEXT NOT NULL,
    model_name            TEXT NOT NULL,
    predicted_class       TEXT NOT NULL,
    inputs_json           TEXT NOT NULL,
    defaulted_fields_json TEXT NOT NULL,
    decision_path_json    TEXT,
    explanation           TEXT,
    explanation_source    TEXT,
    prompt_version        TEXT
)
"""


def _connect(db_path=None):
    conn = sqlite3.connect(db_path or DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_path=None):
    """Create the table if it does not exist yet. Safe to call on every run."""
    with closing(_connect(db_path)) as conn, conn:
        conn.execute(_SCHEMA)


def save_prediction(
    *,
    model_name,
    predicted_class,
    inputs,
    defaulted_fields,
    decision_path=None,
    explanation=None,
    explanation_source=None,
    prompt_version=None,
    db_path=None,
):
    """Insert one prediction row and return its id."""
    with closing(_connect(db_path)) as conn, conn:
        cur = conn.execute(
            """
            INSERT INTO predictions (
                created_at, model_name, predicted_class, inputs_json,
                defaulted_fields_json, decision_path_json, explanation,
                explanation_source, prompt_version
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                datetime.now(timezone.utc).isoformat(timespec="seconds"),
                model_name,
                str(predicted_class),
                json.dumps(inputs),
                json.dumps(list(defaulted_fields)),
                json.dumps(decision_path) if decision_path is not None else None,
                explanation,
                explanation_source,
                prompt_version,
            ),
        )
        return cur.lastrowid


def get_recent(limit=20, db_path=None):
    """Return the newest rows first, as a list of plain dicts."""
    with closing(_connect(db_path)) as conn:
        rows = conn.execute(
            "SELECT * FROM predictions ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()

    result = []
    for row in rows:
        item = dict(row)
        item["inputs"] = json.loads(item.pop("inputs_json"))
        item["defaulted_fields"] = json.loads(item.pop("defaulted_fields_json"))
        path = item.pop("decision_path_json")
        item["decision_path"] = json.loads(path) if path else []
        result.append(item)
    return result