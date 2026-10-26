"""Export KPI tables (for Power BI / Tableau) and render a static KPI dashboard PNG."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def export_kpis(tables: dict, out_dir="outputs/bi_exports"):
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    for name, df in tables.items():
        df.to_csv(out / f"{name}.csv", index=False)
    return sorted(p.name for p in out.glob("*.csv"))


def render_dashboard(k: dict, path="outputs/kpi_dashboard.png"):
    fig, ax = plt.subplots(2, 3, figsize=(17, 9))
    fig.suptitle("Claims Analytics & AI — Component KPI Dashboard", fontsize=16, fontweight="bold")

    m = k["monthly_trend"]
    ax[0, 0].plot(m["month"], m["rejection_rate_pct"], marker="o")
    ax[0, 0].set_title("Rejection rate by month (%)"); ax[0, 0].set_xticks(range(0, len(m), 4)); ax[0, 0].set_xticklabels(m["month"][::4], rotation=45)

    r = k["rejection_by_reason"]
    ax[0, 1].barh(r["rejection_reason"], r["n_claims"], color="#c0504d")
    ax[0, 1].set_title("Rejections by reason"); ax[0, 1].invert_yaxis()

    a = k["anomalies"].head(10)
    ax[0, 2].barh(a.index[::-1], a["anomaly_score"][::-1],
                  color=["#c0504d" if s else "#4f81bd" for s in a["is_seeded_fwa"][::-1]])
    ax[0, 2].set_title("Top-10 anomalous providers (red = true FWA seed)")

    imp = k["feature_importance"].head(7)
    ax[1, 0].barh(imp.index[::-1], imp.values[::-1], color="#4f81bd")
    ax[1, 0].set_title(f"Rejection model drivers (AUC {k['model_metrics']['roc_auc']:.3f})")

    b = k["bottlenecks"]
    ax[1, 1].bar(b["activity"], b["mean"], color="#9bbb59")
    ax[1, 1].set_title("Mean wait before activity (hours)"); ax[1, 1].tick_params(axis="x", rotation=30)

    e = k["nlp_eval"]
    ax[1, 2].bar(e["field"], e["recall_when_present"], color="#8064a2")
    ax[1, 2].set_ylim(0, 1.05)
    ax[1, 2].set_title(f"NLP extraction recall (intent F1 {k['intent_f1']:.2f})")

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    Path(path).parent.mkdir(exist_ok=True, parents=True)
    fig.savefig(path, dpi=130); plt.close(fig)
    return path
