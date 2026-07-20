"""Locate genuine panel datasets in the Rdatasets collection."""

from __future__ import annotations

import io
import json
from pathlib import Path

import pandas as pd
import requests

INDEX = "https://vincentarelbundock.github.io/Rdatasets/datasets.csv"
BASE = "https://vincentarelbundock.github.io/Rdatasets/csv"

PACKAGES = {
    "plm", "AER", "Ecdat", "wooldridge", "pder", "nlme", "lme4",
    "survival", "geepack", "mice", "carData", "HSAUR", "MEMSS",
    "Zelig", "panelr", "causaldata",
}
MIN_ROWS, MAX_ROWS = 100, 200_000
MIN_ENTITIES, MIN_PERIODS = 20, 3


def looks_like_time(name: str) -> bool:
    return any(
        token in name.lower()
        for token in ("year", "time", "period", "wave", "quarter", "month",
                      "week", "date", "visit", "occasion", "t")
    )


def find_structure(df: pd.DataFrame) -> tuple[str, str] | None:
    n = len(df)
    time_candidates = [
        c for c in df.columns
        if looks_like_time(c) and MIN_PERIODS <= df[c].nunique() <= n // MIN_ENTITIES
    ]
    for time_col in sorted(time_candidates, key=lambda c: -df[c].nunique()):
        n_periods = df[time_col].nunique()
        for entity_col in df.columns:
            if entity_col == time_col:
                continue
            n_entities = df[entity_col].nunique(dropna=False)
            if not (MIN_ENTITIES <= n_entities < n):
                continue
            cell_sizes = df.groupby([entity_col, time_col], dropna=False).size()
            if cell_sizes.max() > 1:
                continue
            counts = df[entity_col].value_counts()
            if (counts >= MIN_PERIODS).sum() < MIN_ENTITIES:
                continue
            return entity_col, time_col
    return None


def main() -> None:
    index = pd.read_csv(INDEX)
    shortlist = index[
        index["Package"].isin(PACKAGES)
        & index["Rows"].between(MIN_ROWS, MAX_ROWS)
        & (index["Cols"] >= 4)
    ]
    print(f"screening {len(shortlist)} datasets from {len(PACKAGES)} packages\n")

    found, session = [], requests.Session()
    for _, row in shortlist.iterrows():
        url = f"{BASE}/{row['Package']}/{row['Item']}.csv"
        try:
            resp = session.get(url, timeout=30)
            if resp.status_code != 200:
                continue
            df = pd.read_csv(io.StringIO(resp.text), low_memory=False)
        except Exception:
            continue

        df = df.drop(columns=[c for c in df.columns if c.lower() in ("unnamed: 0", "rownames")],
                     errors="ignore")
        structure = find_structure(df)
        if structure is None:
            continue

        entity_col, time_col = structure
        found.append({
            "package": row["Package"], "item": row["Item"],
            "title": str(row["Title"])[:70], "url": url,
            "entity_col": entity_col, "time_col": time_col,
            "rows": int(len(df)), "columns": int(df.shape[1]),
            "entities": int(df[entity_col].nunique()),
            "periods": int(df[time_col].nunique()),
        })
        print(f"  {row['Package']}/{row['Item']:<22} "
              f"{entity_col} x {time_col}  "
              f"({len(df):,} rows, {df[entity_col].nunique():,} entities)")

    out = Path(__file__).resolve().parents[1] / "results" / "panels.json"
    out.write_text(json.dumps(found, indent=2))
    print(f"\n{len(found)} panels found -> {out}")


if __name__ == "__main__":
    main()
