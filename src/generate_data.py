"""Generate synthetic, deliberately messy healthcare claims data (no real PHI)."""
import numpy as np
import pandas as pd
from pathlib import Path

SEED = 42
SPECIALTIES = ["Cardiology", "Orthopedics", "General Practice", "Radiology", "Dermatology", "Oncology"]
PROCEDURES = {  # code: (base cost, needs_auth)
    "99213": (120, 0), "99214": (180, 0), "93000": (90, 0), "71046": (150, 0),
    "27447": (9000, 1), "70553": (1400, 1), "96413": (2500, 1), "11100": (200, 0),
}
STATES = ["TN", "KA", "MH", "DL", "AP", "TG"]


def make_providers(rng, n=200, n_fwa=8):
    df = pd.DataFrame({
        "provider_id": [f"P{i:04d}" for i in range(n)],
        "specialty": rng.choice(SPECIALTIES, n),
        "state": rng.choice(STATES, n),
    })
    df["is_fwa_seed"] = 0
    df.loc[rng.choice(n, n_fwa, replace=False), "is_fwa_seed"] = 1
    return df


def make_members(rng, n=3000):
    start = pd.Timestamp("2023-01-01")
    enroll = start + pd.to_timedelta(rng.integers(0, 365, n), unit="D")
    term = enroll + pd.to_timedelta(rng.integers(200, 900, n), unit="D")
    return pd.DataFrame({
        "member_id": [f"M{i:05d}" for i in range(n)],
        "dob": pd.Timestamp("1950-01-01") + pd.to_timedelta(rng.integers(0, 365 * 55, n), unit="D"),
        "gender": rng.choice(["M", "F"], n),
        "enroll_date": enroll, "term_date": term,
    })


def make_claims(rng, providers, members, n=20000):
    codes = list(PROCEDURES)
    p_idx = rng.integers(0, len(providers), n)
    m_idx = rng.integers(0, len(members), n)
    proc = rng.choice(codes, n, p=[.28, .2, .12, .1, .03, .04, .03, .2])
    base = np.array([PROCEDURES[c][0] for c in proc], dtype=float)
    needs_auth = np.array([PROCEDURES[c][1] for c in proc])
    fwa = providers["is_fwa_seed"].values[p_idx]
    billed = base * rng.lognormal(0, 0.25, n) * np.where(fwa == 1, rng.uniform(1.8, 3.0, n), 1.0)
    enr = pd.to_datetime(members["enroll_date"].values[m_idx])
    term = pd.to_datetime(members["term_date"].values[m_idx])
    span = (term - enr).days.values
    inside = (rng.random(n) * span).astype(int)             # ~95% of claims inside coverage
    outside = np.where(rng.random(n) < 0.5, -rng.integers(1, 60, n), span + rng.integers(1, 60, n))
    svc = enr + pd.to_timedelta(np.where(rng.random(n) < 0.05, outside, inside), unit="D")
    has_auth = np.where(needs_auth == 1, rng.random(n) > 0.25, 1)
    eligible = np.asarray((svc >= enr) & (svc <= term))
    p_rej = 0.04 + 0.45 * ((needs_auth == 1) & (has_auth == 0)) + 0.6 * (~eligible) + 0.05 * (billed > 3000)
    rejected = rng.random(n) < np.clip(p_rej, 0, 0.95)
    reason = np.where(~eligible, "NOT_ELIGIBLE", np.where((needs_auth == 1) & (has_auth == 0), "MISSING_AUTH",
                      rng.choice(["DUPLICATE_CLAIM", "CODING_ERROR"], n)))
    df = pd.DataFrame({
        "claim_id": [f"C{i:06d}" for i in range(n)],
        "member_id": members["member_id"].values[m_idx],
        "provider_id": providers["provider_id"].values[p_idx],
        "service_date": svc, "procedure_code": proc,
        "billed_amount": billed.round(2), "has_auth": has_auth.astype(int),
        "status": np.where(rejected, "REJECTED", "PAID"),
        "rejection_reason": np.where(rejected, reason, None),
    })
    df["allowed_amount"] = np.where(df.status == "PAID", (df.billed_amount * 0.8).round(2), 0.0)
    return df


def inject_mess(rng, claims, providers):
    """Add real-world data problems that the data-quality layer must catch."""
    c, p = claims.copy(), providers.copy()
    n = len(c)
    c.loc[rng.choice(n, 150, replace=False), "billed_amount"] = np.nan
    c.loc[rng.choice(n, 40, replace=False), "billed_amount"] *= -1
    c.loc[rng.choice(n, 80, replace=False), "procedure_code"] = None
    c = pd.concat([c, c.sample(120, random_state=1)], ignore_index=True)  # duplicate rows
    c.loc[rng.choice(len(c), 60, replace=False), "member_id"] = "M99999"  # orphan members
    c["service_date"] = c["service_date"].dt.strftime("%Y-%m-%d")
    bad = rng.choice(len(c), 50, replace=False)
    c.loc[bad, "service_date"] = "31/02/2024"  # invalid dates
    p["specialty"] = p["specialty"].where(rng.random(len(p)) > 0.15, p["specialty"].str.upper() + " ")
    return c, p


TEMPLATES = {
    "claim_status": ["Hi, I want to check the status of claim {cid}, billed {amt}.",
                     "Calling about claim {cid} from {dt}. Has it been processed yet?"],
    "rejection_dispute": ["My claim {cid} for {amt} was rejected and I don't understand why.",
                          "Claim {cid} got denied on {dt}. I need to appeal this."],
    "enrollment": ["I need to add my spouse to my plan, effective {dt}.",
                   "Can you confirm my coverage is active as of {dt}?"],
    "provider_complaint": ["The provider charged me {amt} on {dt} which seems far too high.",
                           "I was billed {amt} by the clinic for claim {cid}, this looks wrong."],
}


def make_transcripts(rng, claims, n=600):
    rows = []
    intents = list(TEMPLATES)
    for i in range(n):
        intent = intents[rng.integers(0, len(intents))]
        cl = claims.iloc[rng.integers(0, len(claims))]
        amt = f"${rng.integers(50, 9000):,}"
        dt = (pd.Timestamp("2024-01-01") + pd.Timedelta(days=int(rng.integers(0, 600)))).strftime("%d %b %Y")
        tpl = TEMPLATES[intent][rng.integers(0, 2)]
        body = tpl.format(cid=cl.claim_id, amt=amt, dt=dt)
        rows.append({"call_id": f"T{i:04d}", "transcript": "Agent: Thank you for calling. Member: " + body,
                     "true_intent": intent,
                     "true_claim_id": cl.claim_id if "{cid}" in tpl else None,
                     "true_amount": amt if "{amt}" in tpl else None,
                     "true_date": dt if "{dt}" in tpl else None})
    return pd.DataFrame(rows)


def make_event_log(rng, claims, n_cases=2500):
    ids = claims["claim_id"].drop_duplicates().sample(n_cases, random_state=3).tolist()
    rows = []
    for cid in ids:
        t = pd.Timestamp("2024-03-01") + pd.Timedelta(hours=int(rng.integers(0, 2000)))
        path = ["Received", "Validated", "Adjudicated", "Paid"]
        if rng.random() < 0.22:
            path = ["Received", "Validated", "Pended", "Validated", "Adjudicated", "Paid"]  # rework
        if rng.random() < 0.08:
            path.insert(path.index("Adjudicated") + 1, "Manual Override")
        for act in path:
            wait = {"Pended": 40, "Manual Override": 30, "Adjudicated": 12}.get(act, 4)
            t += pd.Timedelta(hours=float(rng.exponential(wait)))
            rows.append({"case_id": cid, "activity": act, "timestamp": t})
    return pd.DataFrame(rows)


def main(out="data"):
    rng = np.random.default_rng(SEED)
    out = Path(out); out.mkdir(exist_ok=True)
    providers, members = make_providers(rng), make_members(rng)
    claims = make_claims(rng, providers, members)
    dirty_claims, dirty_prov = inject_mess(rng, claims, providers)
    dirty_claims.to_csv(out / "claims_raw.csv", index=False)
    dirty_prov.to_csv(out / "providers_raw.csv", index=False)
    members.to_csv(out / "enrollment_raw.csv", index=False)
    make_transcripts(rng, claims).to_csv(out / "call_transcripts.csv", index=False)
    make_event_log(rng, claims).to_csv(out / "workflow_events.csv", index=False)
    print(f"Generated raw data in {out}/")


if __name__ == "__main__":
    main()
