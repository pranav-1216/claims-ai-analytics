import pandas as pd
import numpy as np
from src import data_quality as dq, nlp_transcripts as nlp, process_mining as pm, models


def _claims():
    return pd.DataFrame({
        "claim_id": ["C1", "C1", "C2", "C3"], "member_id": ["M1", "M1", "M2", "MX"],
        "provider_id": ["P1"] * 4, "service_date": ["2024-01-01", "2024-01-01", "31/02/2024", "2024-02-01"],
        "procedure_code": ["99213", "99213", None, "99214"], "billed_amount": [100.0, 100.0, -5.0, np.nan],
        "status": ["PAID"] * 4, "allowed_amount": [80.0, 80.0, 0.0, 0.0],
    })


def test_dq_flags_each_problem():
    r = dq.check_claims(_claims(), {"M1", "M2"}, {"P1"}).to_frame().set_index("check")
    assert r.loc["duplicate_claim_id", "failed_rows"] == 1
    assert r.loc["negative_billed_amount", "failed_rows"] == 1
    assert r.loc["missing_billed_amount", "failed_rows"] == 1
    assert r.loc["unparseable_service_date", "failed_rows"] == 1
    assert r.loc["orphan_member_id", "failed_rows"] == 1


def test_clean_claims_keeps_only_valid_rows():
    out = dq.clean_claims(_claims(), {"M1", "M2"})
    assert list(out["claim_id"]) == ["C1"]


def test_nlp_extraction():
    f = nlp.extract_fields("Agent: Hi. Member: My claim C000123 for $1,250 was denied on 05 Mar 2024.")
    assert (f["claim_id"], f["amount"], f["date"]) == ("C000123", "$1,250", "05 Mar 2024")
    assert nlp.extract_fields("Member: hello there")["claim_id"] is None


def test_process_mining_rework_and_override():
    log = pd.DataFrame({
        "case_id": ["a"] * 4 + ["b"] * 3,
        "activity": ["Received", "Validated", "Validated", "Paid", "Received", "Manual Override", "Paid"],
        "timestamp": pd.date_range("2024-01-01", periods=7, freq="h"),
    })
    s = pm.rework_and_overrides(pm.prepare(log))
    assert s["rework_rate_pct"] == 50.0 and s["override_rate_pct"] == 50.0


def test_risk_tiers():
    t = models.risk_tiers(pd.Series([0.05, 0.3, 0.9]))
    assert list(t) == ["Low", "Medium", "High"]
