"""Reproduce every quantitative claim in the manuscript."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from paneldx import (
    detect_counters,
    discover_keys,
    persistence_baseline,
    target_leakage,
    validate_key,
)
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupKFold

POSITIONAL_KEY = "physician_id"
RECOVERED_KEY = ["Disease", "Opening time"]
TIME_COL = "time"
PUBLISHED_MAE = 0.0942


def zscore(values: pd.Series | np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype="float64")
    return (arr - np.nanmean(arr)) / (np.nanstd(arr) + 1e-8)


def load(path: Path) -> pd.DataFrame:
    df = pd.read_excel(path, sheet_name="Sheet1")
    rows_per_period = len(df) // df[TIME_COL].nunique()
    df[POSITIONAL_KEY] = ((df["Serial number"] - 1) % rows_per_period) + 1
    return df


def add_published_variables(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["log_patients"] = np.log1p(df["Total patients"])
    df["log_visits"] = np.log1p(df["Total visits"])
    df["log_gifts"] = np.log1p(df["Total Gifts"].fillna(0))
    df["gifts_per_visit"] = df["Total Gifts"] / (df["Total visits"] + 1)
    df["inv_rank"] = 1 / (df["Recommended order"] + 1)
    df["article_engagement"] = df["Popular Science Zone"] / (df["Total Articles"] + 1)
    df["PopIdx"] = np.nanmean(
        np.vstack([
            zscore(df["log_visits"]),
            zscore(df["log_gifts"]),
            zscore(df["Patient recommendation"]),
            zscore(df["inv_rank"]),
        ]),
        axis=0,
    )
    return df


def corrected_panel(df: pd.DataFrame) -> pd.DataFrame:
    panel = df[df["Opening time"].notna()].copy()
    panel["pid"] = panel["Disease"].astype(str) + "|" + panel["Opening time"].astype(str)

    per_cell = panel.groupby(["pid", TIME_COL]).size()
    unambiguous = per_cell.groupby("pid").max()
    panel = panel[panel["pid"].isin(unambiguous[unambiguous == 1].index)]

    periods = panel.groupby("pid")[TIME_COL].nunique()
    complete = panel[panel["pid"].isin(periods[periods == panel[TIME_COL].nunique()].index)]
    return complete.sort_values(["pid", TIME_COL])


def section_defects(raw: pd.DataFrame, df: pd.DataFrame) -> dict:
    positional = validate_key(raw, POSITIONAL_KEY, TIME_COL)
    recovered = validate_key(raw, RECOVERED_KEY, TIME_COL)
    discovered = discover_keys(raw, TIME_COL, max_columns=2, top_k=3)

    invariant_breaches = {
        column: float((df.groupby(POSITIONAL_KEY)[column].nunique(dropna=False) > 1).mean())
        for column in ("Opening time", "gender", "Rank")
    }

    violation_rates = {}
    for label, key_cols in (("positional", [POSITIONAL_KEY]), ("recovered", RECOVERED_KEY)):
        ordered = raw.dropna(subset=key_cols).copy()
        ordered["_e"] = ordered[key_cols].astype(str).agg("|".join, axis=1)
        ordered = ordered.sort_values(["_e", TIME_COL])
        steps = ordered.groupby("_e")["Total visits"].diff().dropna()
        violation_rates[label] = float((steps < 0).mean())

    counters = detect_counters(raw, RECOVERED_KEY, TIME_COL)
    features = [
        "log_patients", "log_visits", "log_gifts",
        "gifts_per_visit", "inv_rank", "article_engagement",
    ]
    leakage = target_leakage(df, "PopIdx", features)

    return {
        "positional_key": {
            "evidence_frac": positional.evidence_frac,
            "columns_explained": int(positional.evidence),
            "columns_tested": positional.n_usable_cols,
            "verdict": positional.verdict,
            "invariant_breach_rate": invariant_breaches,
        },
        "recovered_key": {
            "evidence_frac": recovered.evidence_frac,
            "columns_explained": int(recovered.evidence),
            "columns_tested": recovered.n_usable_cols,
            "entities": recovered.n_entities,
            "coverage": recovered.coverage,
            "verdict": recovered.verdict,
            "invariants": recovered.invariant_cols,
            "counters": recovered.monotone_cols,
        },
        "blind_discovery": [
            {"key": list(report.key),
             "evidence_frac": report.evidence_frac,
             "coverage": report.coverage}
            for report in discovered
        ],
        "counter_violation_rate": violation_rates,
        "counters": {
            "detected": counters.counters,
            "autocorrelation": counters.autocorrelation,
        },
        "target_leakage": {
            "r2": leakage.r2,
            "verdict": leakage.verdict,
            "top_contributors": [
                {"feature": name, "weight": weight}
                for name, weight in leakage.top_contributors
            ],
        },
    }


def section_concealment(df: pd.DataFrame) -> dict:
    under_broken = persistence_baseline(df, POSITIONAL_KEY, TIME_COL, "PopIdx")
    under_recovered = persistence_baseline(df, RECOVERED_KEY, TIME_COL, "PopIdx")

    return {
        "under_positional_key": {
            "mae": under_broken.persistence_mae,
            "r2": under_broken.persistence_r2,
            "autocorrelation": under_broken.target_autocorrelation,
            "pairs": under_broken.n_pairs,
        },
        "under_recovered_key": {
            "mae": under_recovered.persistence_mae,
            "r2": under_recovered.persistence_r2,
            "autocorrelation": under_recovered.target_autocorrelation,
            "pairs": under_recovered.n_pairs,
        },
        "published_model_mae": PUBLISHED_MAE,
        "published_model_worse_by": PUBLISHED_MAE / under_recovered.persistence_mae,
    }


def section_satisfaction(panel: pd.DataFrame) -> dict:
    panel = panel.copy()
    panel["tenure_years"] = (
        pd.Timestamp("2023-01-01") - pd.to_datetime(panel["Opening time"])
    ).dt.days / 365.25
    for source, flow in (("Total visits", "new_visits"),
                         ("Total patients", "new_patients"),
                         ("Total Gifts", "new_gifts")):
        panel[flow] = panel.groupby("pid")[source].diff().clip(lower=0)

    final_period = panel[panel[TIME_COL] == panel[TIME_COL].max()].copy()
    final_period = final_period[final_period["Patient recommendation"].notna()]

    numeric_features = [
        "Total patients", "Total visits", "Total Articles",
        "Post-diagnosis evaluation", "Thoughtful Gifts", "Total Gifts",
        "Popular Science Zone", "Medical consultation records",
        "Patient votes", "price", "tenure_years",
        "new_visits", "new_patients", "new_gifts",
    ]
    design = final_period[numeric_features].copy()
    for column in ("Disease", "Rank", "gender"):
        design = design.join(
            pd.get_dummies(final_period[column].astype(str), prefix=column).astype(float)
        )
    design = design.loc[:, ~design.columns.duplicated()].fillna(-1)

    target = final_period["Patient recommendation"].to_numpy(dtype="float64")
    groups = final_period["Disease"].to_numpy()

    import lightgbm as lgb

    predictions = np.zeros(len(design))
    for train_idx, test_idx in GroupKFold(n_splits=4).split(design, target, groups):
        model = lgb.LGBMRegressor(
            n_estimators=500, learning_rate=0.05, num_leaves=31,
            verbose=-1, random_state=0,
        )
        model.fit(design.iloc[train_idx], target[train_idx])
        predictions[test_idx] = model.predict(design.iloc[test_idx])

    mean_only = np.full_like(target, target.mean())
    return {
        "n": int(len(design)),
        "n_features": int(design.shape[1]),
        "target_sd": float(target.std()),
        "mean_baseline_mae": float(mean_absolute_error(target, mean_only)),
        "model_mae": float(mean_absolute_error(target, predictions)),
        "model_r2": float(r2_score(target, predictions)),
        "improvement_over_mean": float(
            1 - mean_absolute_error(target, predictions) / mean_absolute_error(target, mean_only)
        ),
        "validation": "GroupKFold by disease area; each fold scored on an unseen specialty",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument(
        "--out", type=Path,
        default=Path(__file__).resolve().parents[1] / "results" / "findings.json",
    )
    args = parser.parse_args()

    raw_original = load(args.data)
    raw = add_published_variables(raw_original)
    panel = corrected_panel(raw)

    findings = {
        "dataset": {
            "rows": int(len(raw)),
            "columns": int(raw.shape[1]),
            "periods": int(raw[TIME_COL].nunique()),
            "disease_areas": int(raw["Disease"].nunique()),
            "rows_missing_opening_time": int(raw["Opening time"].isna().sum()),
        },
        "corrected_panel": {
            "physicians": int(panel["pid"].nunique()),
            "rows": int(len(panel)),
            "retained_fraction": float(len(panel) / len(raw)),
        },
        "defects": section_defects(raw_original, raw),
        "concealment": section_concealment(raw),
        "satisfaction_task": section_satisfaction(panel),
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(findings, indent=2, default=float))

    d, c, s = findings["defects"], findings["concealment"], findings["satisfaction_task"]
    print(f"\ndataset            {findings['dataset']['rows']:,} rows x "
          f"{findings['dataset']['columns']} cols, "
          f"{findings['dataset']['periods']} periods")
    print(f"corrected panel    {findings['corrected_panel']['physicians']:,} physicians, "
          f"{findings['corrected_panel']['rows']:,} rows")
    print("\nDEFECT 1  entity linkage")
    print(f"  positional key   {d['positional_key']['evidence_frac']:.0%} of columns explained")
    print(f"  recovered key    {d['recovered_key']['evidence_frac']:.0%}, "
          f"{d['recovered_key']['entities']:,} physicians")
    print(f"  blind search #1  {' + '.join(d['blind_discovery'][0]['key'])}")
    print("\nDEFECT 2  cumulative features")
    for name in d["counters"]["detected"][:4]:
        print(f"  {name:<32} rho={d['counters']['autocorrelation'].get(name, float('nan')):.3f}")
    print("\nDEFECT 3  target composition")
    print(f"  held-out R2      {d['target_leakage']['r2']:.3f}")
    print(f"  contributors     "
          f"{[c['feature'] for c in d['target_leakage']['top_contributors'][:3]]}")
    print("\nCONCEALMENT")
    print(f"  under positional  MAE={c['under_positional_key']['mae']:.4f}  "
          f"R2={c['under_positional_key']['r2']:.4f}")
    print(f"  under recovered   MAE={c['under_recovered_key']['mae']:.4f}  "
          f"R2={c['under_recovered_key']['r2']:.4f}")
    print(f"  published model   MAE={c['published_model_mae']:.4f}  "
          f"({c['published_model_worse_by']:.1f}x worse than carry-forward)")
    print("\nWHAT THE DATA DOES SUPPORT")
    print(f"  satisfaction      R2={s['model_r2']:.3f}  MAE={s['model_mae']:.4f} "
          f"vs {s['mean_baseline_mae']:.4f} mean-only "
          f"({s['improvement_over_mean']:.0%} better), n={s['n']:,}")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
