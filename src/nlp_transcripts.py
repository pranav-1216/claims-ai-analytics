"""NLP over call transcripts: rule-based entity extraction + TF-IDF intent classifier,
evaluated against human-reviewed labels."""
import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline

CLAIM_RE = re.compile(r"\bC\d{6}\b")
AMOUNT_RE = re.compile(r"\$\d{1,3}(?:,\d{3})*(?:\.\d{2})?")
DATE_RE = re.compile(r"\b\d{1,2} [A-Z][a-z]{2} \d{4}\b")


def extract_fields(text: str) -> dict:
    """Turn unstructured transcript text into structured fields."""
    m_claim, m_amt, m_date = CLAIM_RE.search(text), AMOUNT_RE.search(text), DATE_RE.search(text)
    member_part = text.split("Member:", 1)[-1].strip()
    return {"claim_id": m_claim.group() if m_claim else None,
            "amount": m_amt.group() if m_amt else None,
            "date": m_date.group() if m_date else None,
            "member_utterance": member_part}


def extract_all(df: pd.DataFrame) -> pd.DataFrame:
    return pd.concat([df[["call_id"]], pd.DataFrame([extract_fields(t) for t in df["transcript"]])], axis=1)


def evaluate_extraction(df: pd.DataFrame, extracted: pd.DataFrame) -> pd.DataFrame:
    """Field-level accuracy versus human-reviewed ground truth (None == field absent)."""
    rows = []
    for ext, truth in [("claim_id", "true_claim_id"), ("amount", "true_amount"), ("date", "true_date")]:
        a, b = extracted[ext].fillna("∅"), df[truth].fillna("∅")
        present = b != "∅"
        rows.append({"field": ext, "accuracy": float((a == b).mean()),
                     "recall_when_present": float((a[present] == b[present]).mean()),
                     "false_positives": int(((a != "∅") & ~present).sum())})
    return pd.DataFrame(rows)


def train_intent_classifier(df: pd.DataFrame, seed=0):
    X = df["transcript"].str.split("Member:", n=1).str[-1]
    Xtr, Xte, ytr, yte = train_test_split(X, df["true_intent"], test_size=0.3, stratify=df["true_intent"], random_state=seed)
    pipe = make_pipeline(TfidfVectorizer(ngram_range=(1, 2), min_df=1), LogisticRegression(max_iter=1000))
    pipe.fit(Xtr, ytr)
    pred = pipe.predict(Xte)
    return pipe, {"macro_f1": float(f1_score(yte, pred, average="macro")),
                  "report": classification_report(yte, pred, output_dict=True)}
