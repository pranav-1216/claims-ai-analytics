"""Lightweight process mining on a claims workflow event log (pure pandas)."""
import pandas as pd


def prepare(log: pd.DataFrame) -> pd.DataFrame:
    df = log.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values(["case_id", "timestamp"]).reset_index(drop=True)
    df["next_activity"] = df.groupby("case_id")["activity"].shift(-1)
    df["wait_hours"] = df.groupby("case_id")["timestamp"].diff().dt.total_seconds() / 3600
    return df


def variants(df: pd.DataFrame) -> pd.DataFrame:
    v = df.groupby("case_id")["activity"].apply(lambda s: " → ".join(s)).value_counts().rename_axis("variant").reset_index(name="cases")
    v["pct"] = (100 * v["cases"] / v["cases"].sum()).round(2)
    return v


def directly_follows(df: pd.DataFrame) -> pd.DataFrame:
    e = df.dropna(subset=["next_activity"])
    return e.groupby(["activity", "next_activity"]).size().reset_index(name="count").sort_values("count", ascending=False)


def rework_and_overrides(df: pd.DataFrame) -> dict:
    per_case = df.groupby("case_id")["activity"].apply(list)
    rework = per_case.apply(lambda a: len(a) != len(set(a)))  # any repeated activity
    override = per_case.apply(lambda a: "Manual Override" in a)
    return {"n_cases": int(len(per_case)), "rework_rate_pct": round(float(100 * rework.mean()), 2),
            "override_rate_pct": round(float(100 * override.mean()), 2)}


def bottlenecks(df: pd.DataFrame) -> pd.DataFrame:
    """Mean/median hours waited *before* each activity."""
    return (df.dropna(subset=["wait_hours"]).groupby("activity")["wait_hours"]
            .agg(["count", "mean", "median"]).round(2).sort_values("mean", ascending=False).reset_index())


def throughput(df: pd.DataFrame) -> pd.Series:
    g = df.groupby("case_id")["timestamp"]
    return ((g.max() - g.min()).dt.total_seconds() / 3600).rename("cycle_time_hours")
