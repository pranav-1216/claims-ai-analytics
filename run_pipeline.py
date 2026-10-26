"""End-to-end pipeline: raw data -> DQ -> SQL -> models -> NLP -> process mining -> KPI exports."""
import json
from pathlib import Path
import pandas as pd

from src import generate_data, data_quality as dq, sql_analysis, models, nlp_transcripts as nlp
from src import process_mining as pm, dashboard

DATA, OUT = Path("data"), Path("outputs")


def main():
    if not (DATA / "claims_raw.csv").exists():
        generate_data.main(str(DATA))
    OUT.mkdir(exist_ok=True)

    # 1. Load + profile + data quality
    claims_raw = pd.read_csv(DATA / "claims_raw.csv")
    prov_raw = pd.read_csv(DATA / "providers_raw.csv")
    enroll = pd.read_csv(DATA / "enrollment_raw.csv")
    dq_report = pd.concat([
        dq.check_claims(claims_raw, set(enroll.member_id), set(prov_raw.provider_id)).to_frame(),
        dq.check_providers(prov_raw).to_frame()])
    dq_report.to_csv(OUT / "data_quality_report.csv", index=False)
    dq.profile(claims_raw).to_csv(OUT / "claims_profile.csv")
    claims = dq.clean_claims(claims_raw, set(enroll.member_id))
    providers = dq.clean_providers(prov_raw)
    print(f"[DQ] {len(claims_raw):,} raw -> {len(claims):,} clean claims")

    # 2. SQL analysis
    con = sql_analysis.build_db(claims, providers, enroll, str(OUT / "claims.db"))
    sql = sql_analysis.run_all(con)

    # 3. Models
    feat = models.provider_features(claims, providers)
    scored = models.detect_provider_anomalies(feat)
    scored["is_seeded_fwa"] = providers.set_index("provider_id")["is_fwa_seed"].reindex(scored.index).astype(bool)
    anomaly_eval = models.evaluate_anomalies(scored, providers)
    X, y = models.rejection_features(claims, providers, enroll)
    model, metrics, imp = models.train_rejection_model(X, y)
    claims["rejection_risk"] = model.predict_proba(X)[:, 1]
    claims["risk_tier"] = models.risk_tiers(claims["rejection_risk"])
    tiers = (claims.groupby("risk_tier", observed=True)
             .agg(n_claims=("claim_id", "size"), actual_rejection_rate=("status", lambda s: (s == "REJECTED").mean()))
             .round(4).reset_index())
    print(f"[ML] anomaly precision@10={anomaly_eval['precision_at_k']:.2f} | rejection AUC={metrics['roc_auc']:.3f}")

    # 4. NLP
    calls = pd.read_csv(DATA / "call_transcripts.csv")
    extracted = nlp.extract_all(calls)
    nlp_eval = nlp.evaluate_extraction(calls, extracted)
    _, intent = nlp.train_intent_classifier(calls)
    extracted.to_csv(OUT / "transcripts_structured.csv", index=False)
    print(f"[NLP] intent macro-F1={intent['macro_f1']:.3f}")

    # 5. Process mining
    log = pm.prepare(pd.read_csv(DATA / "workflow_events.csv"))
    pm_summary = pm.rework_and_overrides(log)
    bott = pm.bottlenecks(log)
    print(f"[PM] {pm_summary}")

    # 6. KPI exports + dashboard
    exports = {**sql, "dq_report": dq_report, "anomalies": scored.reset_index(names="provider_id"),
               "risk_tiers": tiers, "feature_importance": imp.rename("importance").rename_axis("feature").reset_index(),
               "nlp_extraction_eval": nlp_eval, "process_variants": pm.variants(log),
               "directly_follows": pm.directly_follows(log), "bottlenecks": bott,
               "claims_scored": claims[["claim_id", "provider_id", "billed_amount", "status", "rejection_risk", "risk_tier"]]}
    files = dashboard.export_kpis(exports)
    dashboard.render_dashboard({"monthly_trend": sql["monthly_trend"], "rejection_by_reason": sql["rejection_by_reason"],
                                "anomalies": scored, "feature_importance": imp, "model_metrics": metrics,
                                "bottlenecks": bott, "nlp_eval": nlp_eval, "intent_f1": intent["macro_f1"]})
    summary = {"rows_raw": len(claims_raw), "rows_clean": len(claims), "anomaly_eval": anomaly_eval,
               "rejection_model": metrics, "intent_macro_f1": intent["macro_f1"], "process": pm_summary,
               "exports": files}
    (OUT / "results_summary.json").write_text(json.dumps(summary, indent=2, default=float))
    print("Done. See outputs/")


if __name__ == "__main__":
    main()
