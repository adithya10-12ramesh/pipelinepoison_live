"""
harness/results_store.py

Durable, append-only persistence for every evaluation record (FR6), plus
aggregation into per-class / per-mitigation success rates with confidence
intervals (FR7).
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Optional

import pandas as pd


class ResultsStore:
    def __init__(self, path: str | Path = "results/verdicts.jsonl"):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, record: dict) -> None:
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def load(self) -> pd.DataFrame:
        if not self.path.exists() or self.path.stat().st_size == 0:
            return pd.DataFrame()
        rows = []
        with open(self.path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return pd.DataFrame(rows)

    def completed_keys(self) -> set:
        """Cells already stored WITHOUT an API error, as
        (payload_id, base_pr_id, mitigation_state, model, task). Used by
        --skip-existing so reruns never duplicate completed work."""
        df = self.load()
        if df.empty:
            return set()
        if "response_error" in df.columns:
            df = df[df["response_error"].isna()]
        cols = ["payload_id", "base_pr_id", "mitigation_state", "model", "task"]
        if any(c not in df.columns for c in cols):
            return set()
        return set(map(tuple, df[cols].astype(str).itertuples(index=False, name=None)))

    def aggregate(self) -> dict[str, pd.DataFrame]:
        """Returns three tables: per_class, per_model, per_mitigation — each
        with attack success rate (using rule_based_success as the primary
        metric) and a Wilson-ish 95% CI.

        IMPORTANT: rows where the API call itself failed (response_error is
        set — e.g. a quota or network error) are excluded from the success
        count and denominator. Counting a failed call as "attack failed" is
        wrong: the model never actually saw the payload, so it's missing
        data, not a defended attack. Each table's `n_errors` column reports
        how many records were excluded this way, so you can tell whether an
        apparent 0% success rate reflects a real defended run or a run that
        mostly hit API errors."""
        df = self.load()
        if df.empty:
            return {
                "per_class": pd.DataFrame(),
                "per_model": pd.DataFrame(),
                "per_mitigation": pd.DataFrame(),
            }

        has_error_col = "response_error" in df.columns
        if has_error_col:
            errored = df[df["response_error"].notna()]
            valid = df[df["response_error"].isna()]
        else:
            errored = df.iloc[0:0]
            valid = df

        def _agg(group_cols: list[str]) -> pd.DataFrame:
            if valid.empty:
                base_cols = group_cols + ["sum", "count", "success_rate", "ci95", "n_errors"]
                return pd.DataFrame(columns=base_cols)

            g = valid.groupby(group_cols)["rule_based_success"].agg(["sum", "count"])
            g["success_rate"] = g["sum"] / g["count"]
            g["ci95"] = g.apply(
                lambda row: _wilson_ci(int(row["sum"]), int(row["count"])), axis=1
            )
            if "output_validation_flagged" in valid.columns:
                # Defended-by-validator view: an attack only counts as having
                # succeeded if the output validator did NOT flag the reply.
                flagged = valid["output_validation_flagged"].fillna(False).astype(bool)
                sav = (valid["rule_based_success"].astype(bool) & ~flagged).astype(int)
                g["sum_after_validation"] = sav.groupby([valid[c] for c in group_cols]).sum()
                g["rate_after_validation"] = g["sum_after_validation"] / g["count"]
            if not errored.empty:
                err_counts = errored.groupby(group_cols).size()
                g["n_errors"] = err_counts.reindex(g.index).fillna(0).astype(int)
            else:
                g["n_errors"] = 0
            return g.reset_index()

        return {
            "per_class": _agg(["class_id"]),
            "per_model": _agg(["model"]),
            "per_mitigation": _agg(["mitigation_state"]),
        }

    def export_csv(self, out_dir: str | Path = "results") -> None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        for name, table in self.aggregate().items():
            if not table.empty:
                table.to_csv(out_dir / f"aggregate_{name}.csv", index=False)


def _wilson_ci(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = successes / n
    denom = 1 + z**2 / n
    centre = p + z**2 / (2 * n)
    margin = z * math.sqrt((p * (1 - p) + z**2 / (4 * n)) / n)
    lower = (centre - margin) / denom
    upper = (centre + margin) / denom
    return (float(round(max(0.0, lower), 4)), float(round(min(1.0, upper), 4)))
