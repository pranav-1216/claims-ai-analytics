"""Analytical models: provider anomaly detection (FWA patterns), rejection-risk scoring, risk tiers."""
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, precision_score, recall_score, average_precision_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

AUTH_CODES = {"27447", "70553", "96413"}


def provider_features(claims: pd.DataFrame, providers: pd.DataFrame) -> pd.DataFrame:
    df = claims.merge(providers[["provider_id", "specialty"]], on="provider_id")
    # Normalise billed amount per procedure so provider mix does not look like fraud.
    med = df.groupby("procedure_code")["billed_amount"].transform("median")
    df["billed_ratio"] = df["billed_amount"] / med
    g = df.groupby("provider_id")
    feat = pd.DataFrame({
        "n_claims": g.size(),
        "mean_billed_ratio": g["billed_ratio"].mean(),
        "p90_billed_ratio": g["billed_ratio"].quantile(0.9),
        "rejection_rate": g["status"].apply(lambda s: (s == "REJECTED").mean()),
        "claims_per_member": g.size() / g["member_id"].nunique(),
    })
    return feat[feat["n_claims"] >= 20]


def detect_provider_anomalies(feat: pd.DataFrame, contamination=0.05, seed=0) -> pd.DataFrame:
    X = StandardScaler().fit_transform(feat)
    iso = IsolationForest(n_estimators=300, contamination=contamination, random_state=seed).fit(X)
    out = feat.copy()
    out["anomaly_score"] = -iso.score_samples(X)  # higher = more anomalous
    out["is_anomaly"] = iso.predict(X) == -1
    return out.sort_values("anomaly_score", ascending=False)


def evaluate_anomalies(scored: pd.DataFrame, providers: pd.DataFrame, k=10) -> dict:
    truth = providers.set_index("provider_id")["is_fwa_seed"].reindex(scored.index).fillna(0)
    topk = scored.head(k).index
    return {"precision_at_k": float(truth.loc[topk].mean()), "k": k,
            "recall_at_k": float(truth.loc[topk].sum() / max(truth.sum(), 1)),
            "n_seeded_fwa": int(truth.sum())}


def rejection_features(claims, providers, enrollment) -> pd.DataFrame:
    e = enrollment.copy()
    for col in ("dob", "enroll_date", "term_date"):
        e[col] = pd.to_datetime(e[col])
    df = claims.merge(e, on="member_id", how="left").merge(providers[["provider_id", "specialty"]], on="provider_id", how="left")
    X = pd.DataFrame({
        "log_billed": np.log1p(df["billed_amount"]),
        "days_since_enroll": (df["service_date"] - df["enroll_date"]).dt.days,
        "days_to_term": (df["term_date"] - df["service_date"]).dt.days,
        "outside_coverage": ((df["service_date"] < df["enroll_date"]) | (df["service_date"] > df["term_date"])).astype(int),
        "needs_auth": df["procedure_code"].isin(AUTH_CODES).astype(int),
        "has_auth": df["has_auth"],
        "age": (df["service_date"] - df["dob"]).dt.days / 365.25,
    })
    X["missing_required_auth"] = ((X.needs_auth == 1) & (X.has_auth == 0)).astype(int)
    return X.fillna(0), (df["status"] == "REJECTED").astype(int)


def train_rejection_model(X, y, seed=0):
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, stratify=y, random_state=seed)
    model = GradientBoostingClassifier(random_state=seed).fit(Xtr, ytr)
    p = model.predict_proba(Xte)[:, 1]
    pred = (p >= 0.5).astype(int)
    metrics = {"roc_auc": roc_auc_score(yte, p), "pr_auc": average_precision_score(yte, p),
               "precision": precision_score(yte, pred), "recall": recall_score(yte, pred),
               "base_rate": float(y.mean()), "n_test": len(yte)}
    imp = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
    return model, metrics, imp


def risk_tiers(scores: pd.Series) -> pd.Series:
    """Risk stratification: Low / Medium / High by score cut-offs."""
    return pd.cut(scores, bins=[-0.01, 0.2, 0.5, 1.0], labels=["Low", "Medium", "High"])
