"""Plain-language explanations for the COPD severity prediction.

Design rule: the trained model predicts, the LLM only explains.

1. get_decision_facts()  - pulls the facts out of the Decision Tree (no LLM).
2. template_explanation() - rule-based explanation, always available.
3. generate_explanation() - asks Gemini to rewrite the facts in friendlier
   words. If there is no API key, the call fails, or Gemini's answer contains
   numbers that were not in the facts, the template is used instead.
"""

import os
import re

try:  # python-dotenv is optional; the key can also be a normal env variable
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

PROMPT_VERSION = "v1"
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash")

# ---------------------------------------------------------------- features
# (label, unit suffix). Codes for gender / smoking / AGEquartiles are NOT
# documented in the dataset, so they are described only as "coded" values.
# TODO: replace with the real meanings once you have the data dictionary.
FEATURE_INFO = {
    "AGE": ("Age", " years"),
    "PackHistory": ("Smoking history", " pack-years"),
    "MWT1": ("Walk test (attempt 1)", ""),
    "MWT2": ("Walk test (attempt 2)", ""),
    "MWT1Best": ("Walk test (best)", ""),
    "FEV1": ("FEV1", " L"),
    "FEV1PRED": ("FEV1 % predicted", "%"),
    "FVC": ("FVC", " L"),
    "FVCPRED": ("FVC % predicted", "%"),
    "CAT": ("CAT score", ""),
    "SGRQ": ("SGRQ score", ""),
    "AGEquartiles": ("Age group (quartile)", ""),
    "gender": ("Gender (coded value)", ""),
    "smoking": ("Smoking status (coded value)", ""),
    "Diabetes": ("Diabetes (coded 0/1)", ""),
    "muscular": ("Muscular condition (coded 0/1)", ""),
    "hypertension": ("Hypertension (coded 0/1)", ""),
    "AtrialFib": ("Atrial fibrillation (coded 0/1)", ""),
}

# (min, max) used by the app to limit number inputs. CAT is 0-40 by definition.
VALID_RANGES = {
    "AGE": (18.0, 110.0),
    "PackHistory": (0.0, 200.0),
    "MWT1": (0.0, 1000.0),
    "MWT2": (0.0, 1000.0),
    "MWT1Best": (0.0, 1000.0),
    "FEV1": (0.0, 8.0),
    "FEV1PRED": (0.0, 150.0),
    "FVC": (0.0, 10.0),
    "FVCPRED": (0.0, 200.0),
    "CAT": (0.0, 40.0),
    "SGRQ": (0.0, 100.0),
    "AGEquartiles": (1.0, 4.0),
    "gender": (0.0, 1.0),
    "smoking": (1.0, 2.0),
    "Diabetes": (0.0, 1.0),
    "muscular": (0.0, 1.0),
    "hypertension": (0.0, 1.0),
    "AtrialFib": (0.0, 1.0),
}

DISCLAIMER = (
    "This is an educational prototype, not a medical diagnosis. Please "
    "discuss your results with a qualified healthcare professional."
)


def _label(feature):
    return FEATURE_INFO.get(feature, (feature, ""))[0]


def _unit(feature):
    return FEATURE_INFO.get(feature, (feature, ""))[1]


def _fmt(feature, number):
    return f"{number:g}{_unit(feature)}"


# ------------------------------------------------------------------ facts
def _tighten(steps):
    """Merge repeated checks on one feature into its tightest bounds,
    e.g. FEV1PRED <= 48 then FEV1PRED <= 30.5 becomes just <= 30.5."""
    order, best = [], {}
    for s in steps:
        key = (s["feature"], s["op"])
        if s["feature"] not in order:
            order.append(s["feature"])
        if key not in best:
            best[key] = s
        elif s["op"] == "<=" and s["threshold"] < best[key]["threshold"]:
            best[key] = s
        elif s["op"] == ">" and s["threshold"] > best[key]["threshold"]:
            best[key] = s
    return [
        best[(f, op)] for f in order for op in ("<=", ">") if (f, op) in best
    ]


def get_decision_facts(model, input_row, feature_names, defaulted_fields):
    """Collect everything the explanation is allowed to talk about.

    input_row: one-row DataFrame in training column order.
    defaulted_fields: features the user left at the training median.
    """
    defaulted = set(defaulted_fields)
    predicted = str(model.predict(input_row)[0])

    steps = []
    tree = getattr(model, "tree_", None)  # only Decision Trees have a path
    if tree is not None:
        for node in model.decision_path(input_row).indices:
            if tree.children_left[node] == tree.children_right[node]:
                continue  # leaf
            feature = feature_names[tree.feature[node]]
            threshold = float(tree.threshold[node])
            value = float(input_row.iloc[0][feature])
            steps.append(
                {
                    "feature": feature,
                    "value": value,
                    "op": "<=" if value <= threshold else ">",
                    "threshold": round(threshold, 2),
                    "defaulted": feature in defaulted,
                }
            )

    return {
        "predicted_class": predicted,
        "decision_path": _tighten(steps),
        "defaulted_fields": sorted(defaulted),
    }


# --------------------------------------------------------------- template
def _group_by_feature(steps):
    """One entry per feature with its lower/upper bound, in path order."""
    grouped = {}
    for s in steps:
        g = grouped.setdefault(
            s["feature"],
            {"feature": s["feature"], "value": s["value"],
             "above": None, "at_or_below": None, "defaulted": s["defaulted"]},
        )
        if s["op"] == "<=":
            g["at_or_below"] = s["threshold"]
        else:
            g["above"] = s["threshold"]
    return list(grouped.values())


def _describe_group(g):
    f = g["feature"]
    if g["above"] is not None and g["at_or_below"] is not None:
        rel = f"above {_fmt(f, g['above'])} and at or below {_fmt(f, g['at_or_below'])}"
    elif g["at_or_below"] is not None:
        rel = f"at or below {_fmt(f, g['at_or_below'])}"
    else:
        rel = f"above {_fmt(f, g['above'])}"
    return f"{_label(f)} was {_fmt(f, g['value'])}, {rel}"


def template_explanation(facts):
    """Rule-based explanation. Never fails and never needs the internet."""
    parts = [
        f"The model placed this record in the {facts['predicted_class']} "
        "COPD severity group."
    ]
    groups = _group_by_feature(facts["decision_path"])
    if groups:
        parts.append(
            "To get there it checked: "
            + "; ".join(_describe_group(g) for g in groups)
            + "."
        )
    parts.extend(
        f"{_label(g['feature'])} was not entered, so the training-data "
        "median was used for it."
        for g in groups
        if g["defaulted"]
    )
    parts.append(DISCLAIMER)
    return " ".join(parts)


# ----------------------------------------------------------------- Gemini
SYSTEM_PROMPT = (
    "You explain the output of a small machine-learning model that predicts "
    "COPD severity class. You will receive FACTS as JSON. Rewrite them as a "
    "short, friendly explanation for a non-expert (at most 120 words, plain "
    "paragraphs, no lists, no headings).\n"
    "Rules:\n"
    "- Use ONLY the facts provided. Do not introduce any number that is not "
    "in the facts.\n"
    "- Do not diagnose, give treatment, medication or lifestyle advice, or "
    "say how serious the condition is for the person.\n"
    "- If a checked feature is marked defaulted=true, say it was not entered "
    "and the training median was used.\n"
    "- End by saying this is an educational prototype and that they should "
    "talk to a qualified healthcare professional."
)


def build_prompt(facts):
    import json

    checks = [
        {
            "feature": _label(g["feature"]),
            "value": g["value"],
            "above": g["above"],
            "at_or_below": g["at_or_below"],
            "defaulted": g["defaulted"],
        }
        for g in _group_by_feature(facts["decision_path"])
    ]
    return "FACTS:\n" + json.dumps(
        {"predicted_class": facts["predicted_class"], "checks": checks}, indent=2
    )


def _call_gemini(system_prompt, user_prompt):
    # Imported here so the app still runs if google-genai is not installed.
    from google import genai
    from google.genai import types

    client = genai.Client()  # reads GEMINI_API_KEY from the environment
    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=user_prompt,
        config=types.GenerateContentConfig(system_instruction=system_prompt),
    )
    return (response.text or "").strip()


_NUMBER = re.compile(r"\d+(?:\.\d+)?")


def _numbers(text):
    return {round(float(n), 2) for n in _NUMBER.findall(text)}


def _invented_numbers(llm_text, facts):
    """Numbers in the LLM text that are not in the facts (guards against
    made-up values). The facts text includes the class name and checks."""
    allowed = _numbers(build_prompt(facts))
    return sorted(_numbers(llm_text) - allowed)


def generate_explanation(facts):
    """Return {"text", "source", "error"}. Always returns usable text."""
    fallback = template_explanation(facts)

    if not (os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")):
        return {
            "text": fallback,
            "source": "template",
            "error": "No GEMINI_API_KEY set, so the built-in explanation is shown.",
        }

    try:
        text = _call_gemini(SYSTEM_PROMPT, build_prompt(facts))
    except Exception as exc:  # network, quota, bad key, SDK missing, ...
        return {
            "text": fallback,
            "source": "template",
            "error": f"Gemini call failed ({type(exc).__name__}: {exc}).",
        }

    if not text:
        return {"text": fallback, "source": "template",
                "error": "Gemini returned an empty answer."}

    invented = _invented_numbers(text, facts)
    if invented:
        return {
            "text": fallback,
            "source": "template",
            "error": f"Gemini's answer used numbers not in the facts {invented}, "
                     "so it was not shown.",
        }

    return {"text": text, "source": f"gemini:{GEMINI_MODEL}", "error": None}