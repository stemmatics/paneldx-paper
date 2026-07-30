"""Measure specificity and sensitivity across public panel datasets."""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from paneldx import validate_key

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "results" / "cache"
REJECTED = "NOT SUPPORTED"
MIN_WITHIN_ENTITY_VARIATION = 1.5


def fetch(entry: dict) -> pd.DataFrame | None:
    CACHE.mkdir(parents=True, exist_ok=True)
    local = CACHE / f"{entry['package']}_{entry['item']}.csv"
    if not local.exists():
        try:
            resp = requests.get(entry["url"], timeout=30)
            if resp.status_code != 200:
                return None
            local.write_text(resp.text)
        except Exception:
            return None
    df = pd.read_csv(local, low_memory=False)
    return df.drop(
        columns=[c for c in df.columns if c.lower() in ("unnamed: 0", "rownames")],
        errors="ignore",
    )


def corrupt(df: pd.DataFrame, entity_col: str, time_col: str) -> pd.DataFrame | None:
    candidates = [
        c for c in df.columns
        if c not in (entity_col, time_col) and pd.api.types.is_numeric_dtype(df[c])
    ]
    varying = [
        c for c in candidates
        if df.groupby(entity_col)[c].nunique().mean() >= MIN_WITHIN_ENTITY_VARIATION
    ]
    if not varying:
        return None

    sort_on = max(varying, key=lambda c: df.groupby(entity_col)[c].nunique().mean())
    out = df.sort_values([time_col, sort_on]).reset_index(drop=True)
    out["position_id"] = out.groupby(time_col).cumcount()

    if out.groupby("position_id")[entity_col].nunique().eq(1).mean() > 0.5:
        return None
    return out.drop(columns=[entity_col])


def null_evidence(df: pd.DataFrame, entity_col: str, time_col: str, seed: int = 0) -> float:
    rng = np.random.default_rng(seed)
    shuffled = df.copy()
    labels = shuffled[entity_col].to_numpy().copy()
    for _, idx in shuffled.groupby(time_col).groups.items():
        positions = shuffled.index.get_indexer(idx)
        block = labels[positions]
        rng.shuffle(block)
        labels[positions] = block
    shuffled[entity_col] = labels
    try:
        return validate_key(shuffled, entity_col, time_col).evidence
    except Exception:
        return float("nan")


def main() -> None:
    panels = json.loads((ROOT / "results" / "panels.json").read_text())
    rows = []

    for entry in panels:
        df = fetch(entry)
        if df is None or entry["entity_col"] not in df.columns:
            continue
        entity_col, time_col = entry["entity_col"], entry["time_col"]

        try:
            true_key = validate_key(df, entity_col, time_col)
        except Exception:
            continue

        record = {
            "dataset": f"{entry['package']}/{entry['item']}",
            "domain": "clinical" if entry["package"] in
                      {"geepack", "HSAUR", "lme4", "nlme", "survival", "mice"} else "other",
            "rows": entry["rows"],
            "entities": entry["entities"],
            "periods": entry["periods"],
            "columns_tested": true_key.n_usable_cols,
            "true_evidence": int(true_key.evidence),
            "true_evidence_frac": true_key.evidence_frac,
            "true_null_evidence": null_evidence(df, entity_col, time_col),
            "true_verdict": true_key.verdict,
            "true_accepted_share_rule": not true_key.verdict.startswith(REJECTED),
        }

        broken = corrupt(df, entity_col, time_col)
        if broken is not None:
            try:
                fake = validate_key(broken, "position_id", time_col)
                record.update({
                    "fake_evidence": int(fake.evidence),
                    "fake_evidence_frac": fake.evidence_frac,
                    "fake_null_evidence": null_evidence(broken, "position_id", time_col),
                    "fake_rejected_share_rule": fake.verdict.startswith(REJECTED),
                })
            except Exception:
                pass

        rows.append(record)
        mark = "ok" if record["true_accepted_share_rule"] else "FP"
        caught = record.get("fake_rejected_share_rule")
        mark += " ok" if caught else (" FN" if caught is False else " --")
        print(f"  {mark}  {record['dataset']:<28} "
              f"true {record['true_evidence']:>2}/{record['columns_tested']:<3} "
              f"(null {record['true_null_evidence']:.0f})   "
              f"fake {record.get('fake_evidence', float('nan')):>4}")

    data = pd.DataFrame(rows)
    fakes = data[data["fake_rejected_share_rule"].notna()]

    def null_rule(evidence, null):
        return (evidence >= 3) & (evidence - null >= 3)

    data["true_accepted_null_rule"] = null_rule(
        data["true_evidence"], data["true_null_evidence"])
    fakes = fakes.assign(fake_rejected_null_rule=~null_rule(
        fakes["fake_evidence"], fakes["fake_null_evidence"]))

    has_signal = data["true_evidence"] > 0
    with_signal = data[has_signal]

    summary = {
        "n_datasets": int(len(data)),
        "n_with_signal": int(has_signal.sum()),
        "n_without_signal": int((~has_signal).sum()),
        "specificity_overall": float(data["true_accepted_share_rule"].mean()),
        "specificity_given_signal": float(with_signal["true_accepted_share_rule"].mean()),
        "sensitivity": float(fakes["fake_rejected_share_rule"].mean()),
        "null_evidence_max": float(data["true_null_evidence"].max()),
        "null_evidence_mean": float(data["true_null_evidence"].mean()),
        "by_domain_given_signal": {
            domain: {
                "n": int(len(group)),
                "specificity": float(group["true_accepted_share_rule"].mean()),
                "median_columns_tested": float(group["columns_tested"].median()),
            }
            for domain, group in with_signal.groupby("domain")
        },
        "n_corruptible": int(len(fakes)),
        "total_rows_audited": int(data["rows"].sum()),
        "total_entities_audited": int(data["entities"].sum()),
        "share_rule": {
            "specificity": float(data["true_accepted_share_rule"].mean()),
            "sensitivity": float(fakes["fake_rejected_share_rule"].mean()),
        },
        "null_rule": {
            "specificity": float(data["true_accepted_null_rule"].mean()),
            "sensitivity": float(fakes["fake_rejected_null_rule"].mean()),
        },
        "median_evidence_true": float(data["true_evidence"].median()),
        "median_null_evidence_true": float(data["true_null_evidence"].median()),
        "median_evidence_fake": float(fakes["fake_evidence"].median()),
        "no_signal_available": int((data["true_evidence"] == 0).sum()),
        "by_domain": {
            domain: {
                "n": int(len(group)),
                "specificity_share": float(group["true_accepted_share_rule"].mean()),
                "specificity_null": float(group["true_accepted_null_rule"].mean()),
            }
            for domain, group in data.groupby("domain")
        },
    }

    out = ROOT / "results" / "evaluation.json"
    out.write_text(json.dumps({"summary": summary, "datasets": data.to_dict("records")},
                              indent=2, default=float))

    print("\n" + "=" * 64)
    print(f"datasets   {summary['n_datasets']}   "
          f"({summary['total_rows_audited']:,} rows, "
          f"{summary['total_entities_audited']:,} entities)")
    print(f"corruptible {summary['n_corruptible']}\n")
    print(f"{'rule':<14}{'specificity':>13}{'sensitivity':>13}")
    for name in ("share_rule", "null_rule"):
        print(f"{name:<14}{summary[name]['specificity']:>12.1%}"
              f"{summary[name]['sensitivity']:>13.1%}")
    print(f"\nno signal available in {summary['no_signal_available']} datasets "
          f"(0 invariants and 0 counters)")
    for domain, stats in summary["by_domain"].items():
        print(f"  {domain:<10} n={stats['n']:<3} "
              f"share {stats['specificity_share']:.0%}  null {stats['specificity_null']:.0%}")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
