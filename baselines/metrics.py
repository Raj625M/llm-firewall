"""Shared scoring helpers (used on Day 3 baselines, the fine-tuned model, and Day 6).

We report counts ("caught 15 of 20"), not just percentages, and a 95% confidence
interval, because our Hinglish/Hindi test sets are small. With 20 attacks, 15 caught
is "75%", but the honest statement is "somewhere between about 53% and 89%".
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% confidence interval for a rate k/n (Wilson method; works for small n)."""
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def summarize(df: pd.DataFrame, pred_col: str = "pred") -> dict:
    """Scores for one group of rows. Needs columns `label` (1 attack / 0 safe) and `pred`."""
    y = df["label"].to_numpy()
    p = df[pred_col].to_numpy()
    attacks = int((y == 1).sum())
    safe = int((y == 0).sum())
    caught = int(((y == 1) & (p == 1)).sum())
    false_alarms = int(((y == 0) & (p == 1)).sum())

    recall = caught / attacks if attacks else float("nan")          # attacks we caught
    fpr = false_alarms / safe if safe else float("nan")             # safe messages we wrongly blocked
    precision = caught / (caught + false_alarms) if (caught + false_alarms) else float("nan")
    missed = attacks - caught
    # F1 = balance of "catch attacks" and "don't block safe messages". Same as 2*P*R/(P+R).
    f1 = 2 * caught / (2 * caught + false_alarms + missed) if (2 * caught + false_alarms + missed) else float("nan")
    lo, hi = wilson(caught, attacks)
    return {
        "rows": len(df),
        "attacks": attacks,
        "caught": caught,
        "recall": recall,
        "recall_ci_low": lo,
        "recall_ci_high": hi,
        "safe": safe,
        "false_alarms": false_alarms,
        "false_alarm_rate": fpr,
        "precision": precision,
        "f1": f1,
    }


def report(df: pd.DataFrame, by: str | list[str], pred_col: str = "pred") -> pd.DataFrame:
    """One row of scores per group (e.g. per language)."""
    rows = []
    for key, part in df.groupby(by, sort=False):
        row = {"slice": key if isinstance(key, str) else " / ".join(map(str, key))}
        row.update(summarize(part, pred_col))
        rows.append(row)
    return pd.DataFrame(rows)


def pretty(table: pd.DataFrame) -> pd.DataFrame:
    """Human-friendly version for printing: 'caught 15/20', '75% (53-89%)'."""
    # A dash means "this slice has no rows of that kind" (e.g. no attacks in a safe-only category).
    out = pd.DataFrame({"slice": table["slice"]})
    out["attacks caught"] = [f"{r.caught}/{r.attacks}" if r.attacks else "-" for r in table.itertuples()]
    out["recall (95% range)"] = [
        f"{r.recall:.0%} ({r.recall_ci_low:.0%}-{r.recall_ci_high:.0%})" if r.attacks else "-"
        for r in table.itertuples()]
    out["safe blocked"] = [f"{r.false_alarms}/{r.safe}" if r.safe else "-" for r in table.itertuples()]
    out["false-alarm rate"] = [f"{r.false_alarm_rate:.1%}" if r.safe else "-" for r in table.itertuples()]
    # F1 only makes sense when the slice has both attacks and safe messages.
    out["F1"] = [f"{r.f1:.2f}" if (r.attacks and r.safe) else "-" for r in table.itertuples()]
    return out
