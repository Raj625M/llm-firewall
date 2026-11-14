"""
Builds the training dataset for the firewall's classifier.

Run from the project root:
    python data/build_dataset.py

What it does, step by step:
  1. Download public datasets from Hugging Face's web API (first time only; cached in data/raw/).
  2. Load our own examples from data/custom/*.csv (Hinglish, Hindi, hard negatives).
  3. Generate indirect-injection documents (data/indirect.py).
  4. Clean: shorten very long texts, remove empty ones, duplicates, and conflicting labels.
  5. Split into train / val / test, keeping each `group` together and keeping the
     label+language mix similar in every split.
  6. Save data/processed/{train,val,test}.csv and a stats.md summary.
"""

import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import warnings
from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (COLUMNS, CUSTOM_DIR, PROCESSED_DIR, RAW_DIR,  # noqa: E402
                    dedupe_key, guess_lang)
from indirect import make_indirect_examples  # noqa: E402

SEED = 42
MAX_CHARS = 2000            # classifiers read ~512 tokens; longer texts are cut to this length
DOLLY_BENIGN_SAMPLE = 1500  # how many normal instructions to take from dolly
DOLLY_CARRIERS = 150        # how many dolly paragraphs to use as indirect-injection carriers


# ---------------------------------------------------------------------------
# Step 1: public datasets
# ---------------------------------------------------------------------------

HF_API = "https://datasets-server.huggingface.co"


def _get_json(url: str, tries: int = 6) -> dict:
    """GET a URL and parse the JSON reply.

    If the server says "too many requests" (HTTP 429) or is busy (5xx), wait and retry.
    The wait doubles each time (10s, 20s, 40s...), or follows the server's own
    Retry-After header when it sends one. This is called exponential backoff.
    """
    for attempt in range(1, tries + 1):
        try:
            with urllib.request.urlopen(url, timeout=60) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 502, 503) and attempt < tries:
                retry_after = e.headers.get("Retry-After")
                wait = int(retry_after) if retry_after and retry_after.isdigit() else 10 * 2 ** (attempt - 1)
                print(f"    server said {e.code}, waiting {wait}s (attempt {attempt}/{tries})")
                time.sleep(wait)
                continue
            raise
        except urllib.error.URLError:
            if attempt < tries:
                time.sleep(10 * attempt)
                continue
            raise


def fetch_rows(dataset: str, max_rows_per_split: int | None = None) -> pd.DataFrame:
    """Download a public Hugging Face dataset through the Dataset Viewer web API.

    Why not the `datasets` library? It depends on `pyarrow`, whose compiled files are
    blocked by Windows Smart App Control on this PC. The web API returns plain JSON,
    so it needs nothing but Python itself. It serves at most 100 rows per request,
    so we page through each split with `offset`.
    """
    q = urllib.parse.quote(dataset, safe="")
    splits = _get_json(f"{HF_API}/splits?dataset={q}")["splits"]
    frames = []
    for sp in splits:
        config, split = sp["config"], sp["split"]
        offset, rows = 0, []
        while True:
            url = (f"{HF_API}/rows?dataset={q}&config={urllib.parse.quote(config)}"
                   f"&split={urllib.parse.quote(split)}&offset={offset}&length=100")
            page = _get_json(url)
            # Skip rows the server had to shorten (very long cells): we want full text only.
            rows += [r["row"] for r in page["rows"] if not r.get("truncated_cells")]
            offset += 100
            total = page.get("num_rows_total", 0)
            if offset >= total or (max_rows_per_split and offset >= max_rows_per_split):
                break
            time.sleep(1)  # be polite to the free API: about one request per second
        frames.append(pd.DataFrame(rows))
    return pd.concat(frames, ignore_index=True)


def _convert_deepset(df: pd.DataFrame) -> pd.DataFrame:
    # deepset/prompt-injections: columns text, label (1 = injection). English + some German.
    return pd.DataFrame({
        "text": df["text"],
        "label": df["label"].astype(int),
        "category": df["label"].astype(int).map({1: "injection", 0: "benign"}),
        "source": "deepset/prompt-injections",
    })


def _convert_jackhhao(df: pd.DataFrame) -> pd.DataFrame:
    # jackhhao/jailbreak-classification: columns prompt, type ("jailbreak" / "benign").
    is_jb = (df["type"] == "jailbreak").astype(int)
    return pd.DataFrame({
        "text": df["prompt"],
        "label": is_jb,
        "category": is_jb.map({1: "jailbreak", 0: "benign"}),
        "source": "jackhhao/jailbreak-classification",
    })


def _convert_dolly(df: pd.DataFrame) -> pd.DataFrame:
    # databricks/databricks-dolly-15k: normal human-written instructions (all benign).
    # We keep the instruction, and the 'context' paragraph as a document carrier.
    instructions = pd.DataFrame({"text": df["instruction"], "label": 0,
                                 "category": "benign", "source": "dolly"})
    ctx = df["context"].fillna("")
    ctx = ctx[(ctx.str.len() > 200) & (ctx.str.len() < 1200)]
    carriers = pd.DataFrame({"text": ctx, "label": 0,
                             "category": "carrier",   # becomes indirect-injection carriers
                             "source": "dolly_context"})
    return pd.concat([instructions, carriers], ignore_index=True)


# (dataset name, cache file, max rows per split, converter)
# Dolly has 15k rows but we only need ~1,500 instructions + ~150 paragraphs: 3,000 is plenty
# and keeps us to ~30 requests, well inside the free API's limits.
PUBLIC_SOURCES = [
    ("deepset/prompt-injections", "deepset.csv", None, _convert_deepset),
    ("jackhhao/jailbreak-classification", "jackhhao.csv", None, _convert_jackhhao),
    ("databricks/databricks-dolly-15k", "dolly.csv", 3000, _convert_dolly),
]


def download_public() -> pd.DataFrame:
    """Download each public dataset once and cache it in its own file in data/raw/.

    One cache file PER dataset means: if one download fails, the others are kept,
    and the next run only retries the one that failed.
    """
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    frames = []
    for name, cache_name, max_rows, convert in PUBLIC_SOURCES:
        cache = RAW_DIR / cache_name
        if cache.exists():
            df = pd.read_csv(cache)
            print(f"  {name}: {len(df)} rows (cached)")
        else:
            try:
                print(f"  {name}: downloading...")
                df = convert(fetch_rows(name, max_rows_per_split=max_rows))
                df.to_csv(cache, index=False, encoding="utf-8")
                print(f"  {name}: {len(df)} rows")
            except Exception as e:
                print(f"  WARNING: could not load {name}: {e}")
                print("           Run the script again later to retry just this one.")
                continue
        frames.append(df)

    if not frames:
        raise SystemExit("No public data could be downloaded. Check your internet connection.")

    public = pd.concat(frames, ignore_index=True)
    # Combined copy, read by generate_hinglish.py.
    public.to_csv(RAW_DIR / "public.csv", index=False, encoding="utf-8")
    return public


# ---------------------------------------------------------------------------
# Step 2: our own examples
# ---------------------------------------------------------------------------

def load_custom() -> pd.DataFrame:
    """Load every CSV in data/custom/ (columns: text, label, lang, category).

    generated_review.csv is special: it holds LLM-made translations that you must
    review. Only rows where you typed 'y' in the `keep` column are used.
    """
    frames = []
    for path in sorted(CUSTOM_DIR.glob("*.csv")):
        df = pd.read_csv(path, encoding="utf-8-sig")
        if "keep" in df.columns:
            df = df[df["keep"].astype(str).str.strip().str.lower() == "y"]
        if df.empty:
            continue
        df["source"] = path.name
        # `group` lets a translation stay in the same split as its original.
        if "group" not in df.columns or df["group"].isna().all():
            df["group"] = None
        frames.append(df)
        print(f"  {path.name}: {len(df)} rows")
    if not frames:
        return pd.DataFrame(columns=COLUMNS)
    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------
# Steps 3-6
# ---------------------------------------------------------------------------

def assemble(public: pd.DataFrame, custom: pd.DataFrame) -> pd.DataFrame:
    # Public data: drop carriers (used separately), sample dolly so benign doesn't swamp attacks.
    carriers = public.loc[public["category"] == "carrier", "text"]
    carriers = carriers.sample(min(DOLLY_CARRIERS, len(carriers)), random_state=SEED).tolist()

    pub = public[public["category"] != "carrier"].copy()
    dolly = pub[pub["source"] == "dolly"]
    pub = pd.concat([
        pub[pub["source"] != "dolly"],
        dolly.sample(min(DOLLY_BENIGN_SAMPLE, len(dolly)), random_state=SEED),
    ])
    pub["lang"] = pub["text"].astype(str).map(guess_lang)

    indirect = pd.DataFrame(make_indirect_examples(carriers, seed=SEED))
    print(f"  generated indirect-injection examples: {len(indirect)}")

    df = pd.concat([pub, custom, indirect], ignore_index=True)

    # Every row needs a group. Rows without one get their own (based on their text).
    df["text"] = df["text"].astype(str).str.strip()
    df["key"] = df["text"].map(dedupe_key)
    df["group"] = df["group"].where(df["group"].notna() & (df["group"].astype(str) != ""), "row-" + df["key"])
    df["label"] = df["label"].astype(int)
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    # Long texts are CUT to MAX_CHARS, not thrown away: many long ones are jailbreak
    # prompts, and the attack part is usually near the start. The classifier only
    # reads the first ~512 tokens anyway.
    too_long = df["text"].str.len() > MAX_CHARS
    df.loc[too_long, "text"] = df.loc[too_long, "text"].str[:MAX_CHARS]
    print(f"  shortened {too_long.sum()} long texts to {MAX_CHARS} characters")

    before = len(df)
    df = df[df["key"].str.len() >= 3]
    print(f"  removed {before - len(df)} empty texts")

    # Same text with BOTH labels = the sources disagree. We can't trust either, so drop it.
    labels_per_key = df.groupby("key")["label"].nunique()
    conflicting = labels_per_key[labels_per_key > 1].index
    df = df[~df["key"].isin(conflicting)]
    print(f"  removed {len(conflicting)} texts that had conflicting labels")

    before = len(df)
    df = df.drop_duplicates(subset="key")
    print(f"  removed {before - len(df)} duplicates")
    return df


def split(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """~70 / 15 / 15 split that keeps groups together and balances label+language.

    StratifiedGroupKFold makes 7 folds where (a) a group never spans two folds and
    (b) each fold has roughly the same mix of the `strata` values. We use one fold as
    test, one as validation, and the other five as train.
    """
    strata = df["label"].astype(str) + "_" + df["lang"]
    folds = StratifiedGroupKFold(n_splits=7, shuffle=True, random_state=SEED)
    # sklearn warns when a language has fewer than 7 examples of a label. That's
    # expected while your Hinglish/Hindi set is still small, so we silence it.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        fold_ids = list(folds.split(df, strata, groups=df["group"]))
    test_idx = fold_ids[0][1]
    val_idx = fold_ids[1][1]
    rest = set(range(len(df))) - set(test_idx) - set(val_idx)
    return {
        "train": df.iloc[sorted(rest)],
        "val": df.iloc[val_idx],
        "test": df.iloc[test_idx],
    }


def write_stats(splits: dict[str, pd.DataFrame]) -> str:
    lines = ["# Dataset statistics", ""]
    for name, part in splits.items():
        lines.append(f"## {name} ({len(part)} rows)")
        table = pd.crosstab(part["lang"], part["label"].map({0: "benign", 1: "malicious"}), margins=True)
        lines.append(table.to_markdown())
        lines.append("")
    all_rows = pd.concat(splits.values())
    lines.append("## By category (all splits)")
    lines.append(all_rows["category"].value_counts().to_frame("rows").to_markdown())
    lines.append("")
    lines.append("## By source (all splits)")
    lines.append(all_rows["source"].value_counts().to_frame("rows").to_markdown())
    return "\n".join(lines)


def check_balance(df: pd.DataFrame) -> None:
    """Warn when a language has more attacks than safe examples.

    If most Hinglish examples are attacks, the model learns the shortcut
    "Hinglish = attack" and blocks normal users. Aim for at least as many benign
    as malicious examples in every language.
    """
    print("\nBalance check (benign : malicious per language)")
    for lang, part in df.groupby("lang"):
        benign, malicious = (part["label"] == 0).sum(), (part["label"] == 1).sum()
        flag = "  <-- WARNING: add more benign examples in this language" if malicious > benign else ""
        print(f"  {lang:<9} {benign:>5} : {malicious:<5}{flag}")


def main():
    print("Step 1: public data")
    public = download_public()
    print("Step 2: custom data")
    custom = load_custom()
    print("Step 3: assemble + generate indirect injections")
    df = assemble(public, custom)
    print("Step 4: clean")
    df = clean(df)
    check_balance(df)
    print("\nStep 5: split")
    splits = split(df)

    # Sanity check: no group and no text appears in two splits.
    for a, b in [("train", "val"), ("train", "test"), ("val", "test")]:
        assert not set(splits[a]["group"]) & set(splits[b]["group"]), f"group leak {a}/{b}"
        assert not set(splits[a]["key"]) & set(splits[b]["key"]), f"text leak {a}/{b}"

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    for name, part in splits.items():
        part[COLUMNS].sample(frac=1, random_state=SEED).to_csv(
            PROCESSED_DIR / f"{name}.csv", index=False, encoding="utf-8-sig")
    stats = write_stats(splits)
    (PROCESSED_DIR / "stats.md").write_text(stats, encoding="utf-8")
    print("\n" + stats)
    print(f"\nSaved to {PROCESSED_DIR}")


if __name__ == "__main__":
    main()
