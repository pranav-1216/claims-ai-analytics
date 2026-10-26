"""Load cleaned tables into SQLite and run the named queries in sql/queries.sql."""
import re
import sqlite3
from pathlib import Path
import pandas as pd


def build_db(claims, providers, enrollment, path=":memory:") -> sqlite3.Connection:
    if path != ":memory:":
        Path(path).unlink(missing_ok=True)  # rebuild fresh each run
    con = sqlite3.connect(path)
    c = claims.copy(); c["service_date"] = c["service_date"].dt.strftime("%Y-%m-%d")
    e = enrollment.copy()
    for col in ("dob", "enroll_date", "term_date"):
        e[col] = pd.to_datetime(e[col]).dt.strftime("%Y-%m-%d")
    c.to_sql("claims", con, index=False)
    providers.to_sql("providers", con, index=False)
    e.to_sql("enrollment", con, index=False)
    con.execute("CREATE INDEX idx_claims_provider ON claims(provider_id)")
    con.execute("CREATE INDEX idx_claims_member ON claims(member_id)")
    return con


def load_queries(sql_path="sql/queries.sql") -> dict:
    text = Path(sql_path).read_text()
    parts = re.split(r"^-- name: (\w+)\s*$", text, flags=re.M)
    return {parts[i]: parts[i + 1].strip() for i in range(1, len(parts), 2)}


def run_all(con, sql_path="sql/queries.sql") -> dict:
    return {name: pd.read_sql_query(q, con) for name, q in load_queries(sql_path).items()}
