"""Reusable data-quality checks and cleaning for claims / enrollment / provider data."""
from dataclasses import dataclass, field
import pandas as pd


@dataclass
class CheckResult:
    name: str
    failed_rows: int
    total_rows: int
    severity: str = "error"

    @property
    def pass_rate(self) -> float:
        return 1.0 if self.total_rows == 0 else 1 - self.failed_rows / self.total_rows


@dataclass
class DQReport:
    table: str
    results: list = field(default_factory=list)

    def add(self, name, mask, severity="error"):
        self.results.append(CheckResult(name, int(mask.sum()), len(mask), severity))

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame([{"table": self.table, "check": r.name, "failed_rows": r.failed_rows,
                              "total_rows": r.total_rows, "pass_rate": round(r.pass_rate, 4),
                              "severity": r.severity} for r in self.results])


def profile(df: pd.DataFrame) -> pd.DataFrame:
    """Column-level profile: dtype, null %, distinct count, example value."""
    return pd.DataFrame({
        "dtype": df.dtypes.astype(str),
        "null_pct": (df.isna().mean() * 100).round(2),
        "n_unique": df.nunique(),
        "example": df.apply(lambda s: s.dropna().iloc[0] if s.notna().any() else None),
    })


def check_claims(claims: pd.DataFrame, member_ids: set, provider_ids: set) -> DQReport:
    r = DQReport("claims")
    r.add("missing_billed_amount", claims["billed_amount"].isna())
    r.add("negative_billed_amount", claims["billed_amount"] < 0)
    r.add("missing_procedure_code", claims["procedure_code"].isna())
    r.add("duplicate_claim_id", claims.duplicated("claim_id", keep="first"))
    r.add("unparseable_service_date", pd.to_datetime(claims["service_date"], format="%Y-%m-%d", errors="coerce").isna())
    r.add("orphan_member_id", ~claims["member_id"].isin(member_ids))
    r.add("orphan_provider_id", ~claims["provider_id"].isin(provider_ids))
    r.add("paid_but_zero_allowed", (claims["status"] == "PAID") & (claims["allowed_amount"] <= 0), "warning")
    return r


def check_providers(prov: pd.DataFrame) -> DQReport:
    r = DQReport("providers")
    r.add("duplicate_provider_id", prov.duplicated("provider_id"))
    r.add("specialty_not_standardized", prov["specialty"] != prov["specialty"].str.strip().str.title(), "warning")
    return r


def clean_claims(claims: pd.DataFrame, member_ids: set) -> pd.DataFrame:
    """Standardize and drop records that fail hard checks. Returns a new frame."""
    df = claims.drop_duplicates("claim_id", keep="first").copy()
    df["service_date"] = pd.to_datetime(df["service_date"], format="%Y-%m-%d", errors="coerce")
    df["procedure_code"] = df["procedure_code"].astype("string")
    ok = (df["billed_amount"].notna() & (df["billed_amount"] > 0) & df["procedure_code"].notna()
          & df["service_date"].notna() & df["member_id"].isin(member_ids))
    return df[ok].reset_index(drop=True)


def clean_providers(prov: pd.DataFrame) -> pd.DataFrame:
    df = prov.drop_duplicates("provider_id").copy()
    df["specialty"] = df["specialty"].str.strip().str.title()
    return df
