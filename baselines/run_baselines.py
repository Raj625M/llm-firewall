"""Day 3 (part 1): baselines.

A "baseline" is the simplest thing that could work. Before training a big model we
measure two simple ones, so that later we can say honestly how much better the big
model really is.

    1. rules          - keyword/pattern filter (baselines/rules.py), no training
    2. tfidf_plain    - counts letter-groups ("n-grams") in the message and learns
                        which ones appear in attacks (logistic regression)
    3. tfidf_weighted - same, but Hinglish/Hindi training rows count 4x, because
                        they are only ~12% of our training data

Everything is trained on data/processed/train.csv and scored on test.csv. We never
change anything after looking at the test numbers.

Run from the project folder:
    python baselines/run_baselines.py

Output: tables in the terminal, plus
    results/baselines.csv     all numbers (commit this one)
    results/errors_test.csv   every mistake on the test set (for you to read; keep local)
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))

import metrics  # noqa: E402
import rules  # noqa: E402

if hasattr(sys.stdout, "reconfigure"):  # make sure Hindi prints correctly on Windows
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

PROCESSED = ROOT / "data" / "processed"
RESULTS = ROOT / "results"
LOW_RESOURCE = {"hi", "hinglish"}
LANG_WEIGHT = 4.0
LANG_ORDER = ["en", "hinglish", "hi", "other"]


# --------------------------------------------------------------------------- data
def origin_of(source: str) -> str:
    """Who/what produced the row. Matters because machine translations are easier
    and less realistic than messages a human wrote."""
    if source == "my_examples.csv":
        return "written by you"
    if source in {"hinglish_seed.csv", "hard_negatives.csv", "my_examples_extra.csv"}:
        return "written by Claude (seed)"
    if source == "generated_review.csv":
        return "machine-translated, you reviewed"
    if source == "generated_indirect":
        return "template-generated"
    return "public dataset"


def load(split: str) -> pd.DataFrame:
    df = pd.read_csv(PROCESSED / f"{split}.csv", encoding="utf-8-sig")
    df["text"] = df["text"].astype(str)
    df["origin"] = df["source"].map(origin_of)
    return df


# ----------------------------------------------------------------------- models
def fit_tfidf(train: pd.DataFrame):
    """Returns (vectorizer, plain_model, weighted_model)."""
    # char_wb = letter groups of length 2-5 inside each word. Works on Hinglish
    # because it does not care about spelling variants like "bhool"/"bhul".
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), sublinear_tf=True,
                          min_df=2, max_features=300_000)
    X = vec.fit_transform(train["text"])
    y = train["label"].to_numpy()

    plain = LogisticRegression(C=10, class_weight="balanced", max_iter=3000).fit(X, y)

    weights = np.where(train["lang"].isin(LOW_RESOURCE), LANG_WEIGHT, 1.0)
    weighted = LogisticRegression(C=10, class_weight="balanced", max_iter=3000).fit(X, y, sample_weight=weights)
    return vec, plain, weighted


def predict_all(df: pd.DataFrame, vec, plain, weighted) -> dict[str, pd.DataFrame]:
    """One copy of df per model, with columns `pred` (0/1) and `score` (0-1, or NaN for rules)."""
    X = vec.transform(df["text"])
    out = {}

    r = df.copy()
    r["matched"] = [", ".join(rules.check(t)) for t in r["text"]]
    r["pred"] = (r["matched"] != "").astype(int)
    r["score"] = np.nan
    out["rules"] = r

    for name, clf in [("tfidf_plain", plain), ("tfidf_weighted", weighted)]:
        d = df.copy()
        d["score"] = clf.predict_proba(X)[:, 1]
        d["pred"] = (d["score"] >= 0.5).astype(int)
        d["matched"] = ""
        out[name] = d
    return out


# ------------------------------------------------------------------- evaluation
def slices(df: pd.DataFrame) -> list[tuple[str, pd.DataFrame]]:
    """All the ways we cut the results: (kind of slice, table)."""
    parts = []

    overall = metrics.report(df.assign(_all="ALL"), "_all")
    parts.append(("overall", overall))

    lang = metrics.report(df, "lang")
    lang["_o"] = lang["slice"].map({l: i for i, l in enumerate(LANG_ORDER)})
    parts.append(("language", lang.sort_values("_o").drop(columns="_o")))

    low = df[df["lang"].isin(LOW_RESOURCE)]
    parts.append(("origin (Hinglish+Hindi only)", metrics.report(low, "origin")))

    # Does the model just use message length as a shortcut? Short attacks are the test.
    length = pd.cut(df["text"].str.len(), bins=[0, 100, 300, 10**9],
                    labels=["short (under 100 chars)", "medium (100-300)", "long (over 300)"]).astype(str)
    parts.append(("length", metrics.report(df.assign(_len=length), "_len")))

    parts.append(("category", metrics.report(df, "category")))
    return parts


def evaluate(preds: dict[str, pd.DataFrame], split: str) -> pd.DataFrame:
    rows = []
    for model, df in preds.items():
        for kind, table in slices(df):
            t = table.copy()
            t.insert(0, "slice_type", kind)
            t.insert(0, "split", split)
            t.insert(0, "model", model)
            rows.append(t)
    return pd.concat(rows, ignore_index=True)


def show(results: pd.DataFrame, kind: str, title: str) -> None:
    part = results[(results["slice_type"] == kind)]
    pretty = metrics.pretty(part)
    pretty.insert(0, "model", part["model"].to_numpy())
    print(f"\n### {title}")
    print(pretty.to_markdown(index=False))


# -------------------------------------------------------------- shortcut checks
TOPIC_WORDS = (r"\b(?:trump|germany|german|deutschland|spd|greens?|merkel|afd|putin|nato|wef|"
               r"burkina|kurz|holland|mercedes|volkswagen)\b")


def shortcut_checks(train: pd.DataFrame) -> None:
    """A model can score well by learning a trick instead of the real skill.
    These quick checks look for tricks hiding in OUR data."""
    print("\n" + "=" * 70)
    print("SHORTCUT CHECKS on the training data")
    print("=" * 70)

    print(f"\n1) Attack share by language (overall {train['label'].mean():.0%}):")
    for lang in LANG_ORDER:
        part = train[train["lang"] == lang]
        if len(part):
            print(f"   {lang:9s} {part['label'].mean():5.0%} attacks   ({len(part)} rows)")

    topic = train["text"].str.lower().str.contains(TOPIC_WORDS, regex=True)
    print(f"\n2) Rows that mention politics/Germany words (trump, germany, spd, greens, merkel ...):")
    print(f"   {int(topic.sum())} rows, {train.loc[topic, 'label'].mean():.0%} of them are attacks "
          f"(vs {train.loc[~topic, 'label'].mean():.0%} for all other rows)")
    print("   -> if that is far above the overall share, the model can learn 'mentions Trump = attack'.")

    print("\n3) How well does message LENGTH alone separate attacks from safe? (0.5 = useless, 1.0 = perfect)")
    for lang in LANG_ORDER:
        part = train[train["lang"] == lang]
        if part["label"].nunique() == 2:
            auc = roc_auc_score(part["label"], part["text"].str.len())
            print(f"   {lang:9s} {auc:.2f}")

    vec = TfidfVectorizer(analyzer="word", min_df=3, sublinear_tf=True,
                          token_pattern=r"[^\s\.,!?;:\(\)\[\]\"'।\-/]+")
    X = vec.fit_transform(train["text"])
    clf = LogisticRegression(C=5, class_weight="balanced", max_iter=3000).fit(X, train["label"])
    names = np.array(vec.get_feature_names_out())
    order = np.argsort(clf.coef_[0])
    print("\n4) Words the model leans on MOST to say 'attack':")
    print("   " + ", ".join(names[order[::-1][:30]]))
    print("   Words it leans on most to say 'safe':")
    print("   " + ", ".join(names[order[:30]]))
    print("   -> good: 'ignore', 'instructions', 'prompt'. Bad: topic words (trump, germany) or names.")


# ------------------------------------------------------------------------ main
def main() -> None:
    train, val, test = load("train"), load("val"), load("test")
    print(f"train {len(train)} | val {len(val)} | test {len(test)} rows")

    vec, plain, weighted = fit_tfidf(train)
    test_preds = predict_all(test, vec, plain, weighted)
    val_preds = predict_all(val, vec, plain, weighted)

    results = pd.concat([evaluate(test_preds, "test"), evaluate(val_preds, "val")], ignore_index=True)
    RESULTS.mkdir(exist_ok=True)
    results.to_csv(RESULTS / "baselines.csv", index=False, encoding="utf-8-sig")

    t = results[results["split"] == "test"]
    print("\n" + "=" * 70)
    print("BASELINE RESULTS ON THE TEST SET")
    print("(recall = share of attacks caught; false-alarm rate = share of SAFE messages wrongly blocked)")
    print("=" * 70)
    show(t, "overall", "Overall")
    show(t, "language", "By language")
    show(t, "origin (Hinglish+Hindi only)", "Hinglish + Hindi rows, by where the row came from")
    show(t, "length", "By message length (a big drop on SHORT messages = the model leans on length)")
    show(t, "category","By category (attack categories show recall; safe categories show false alarms)")

    # every mistake, for you to read
    errs = []
    for model, d in test_preds.items():
        bad = d[d["pred"] != d["label"]].copy()
        bad["error"] = np.where(bad["label"] == 1, "missed attack", "false alarm")
        bad.insert(0, "model", model)
        errs.append(bad[["model", "error", "lang", "origin", "category", "score", "matched", "text"]])
    errs = pd.concat(errs)
    errs["_o"] = errs["lang"].map({l: i for i, l in enumerate(LANG_ORDER)})
    errs = errs.sort_values(["_o", "model", "error"]).drop(columns="_o")
    errs.to_csv(RESULTS / "errors_test.csv", index=False, encoding="utf-8-sig")
    print(f"\nSaved {RESULTS / 'baselines.csv'}")
    print(f"Saved {RESULTS / 'errors_test.csv'}  ({len(errs)} mistakes; open it and read the Hinglish/Hindi ones)")

    shortcut_checks(train)


if __name__ == "__main__":
    main()
