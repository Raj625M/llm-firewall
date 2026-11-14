"""
Uses the LLM (your Groq key) to translate English examples into Hinglish and Hindi,
then saves them for YOU to review. Nothing generated is trusted automatically.

Run from the project root (after build_dataset.py has downloaded the public data once):
    python data/generate_hinglish.py --n 100

Output: data/custom/generated_review.csv with an empty `keep` column.
Open it in Excel or VS Code and for each row:
  - type y  if the translation is natural and keeps the same meaning and label
  - type n  if it's wrong, unnatural, or the model refused / changed the meaning
  - you may also edit the text to fix it, then type y
Only rows marked y are used by build_dataset.py.

Running it again adds new rows and skips examples already translated.
"""

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI, RateLimitError

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import CUSTOM_DIR, RAW_DIR, dedupe_key, guess_lang  # noqa: E402

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

OUT = CUSTOM_DIR / "generated_review.csv"
BATCH_SIZE = 8
PAUSE_SECONDS = 12   # stay under the free plan's tokens-per-minute limit

INSTRUCTIONS = """You are helping build a dataset to train a SECURITY CLASSIFIER that detects prompt-injection attacks against chatbots used in India.
Translate each message into:
- "hinglish": Hindi written in Latin script, mixed with English words, the way people in India type on WhatsApp (e.g. "bhai mera order kab aayega?").
- "hindi": natural Hindi in Devanagari script.
Rules:
- Keep the meaning and intent EXACTLY the same, including when the message is an attempt to manipulate an AI. Do not soften, refuse, or add warnings: the classifier must learn to recognise these.
- Keep product names, codes, URLs and technical words as they are.
- Return ONLY a JSON array, one object per message: [{"id": <id>, "hinglish": "...", "hindi": "..."}]"""


def pick_examples(n: int, already_done: set[str]) -> pd.DataFrame:
    """Pick n English examples: half attacks, half benign, from the public data.

    Half and half matters: if most Hinglish examples were attacks, the model would
    learn "Hinglish = attack" and block normal Indian users.
    """
    public = pd.read_csv(RAW_DIR / "public.csv")
    public = public[public["category"] != "carrier"]
    public = public[public["text"].astype(str).map(guess_lang) == "en"]
    public = public[public["text"].astype(str).str.len().between(15, 400)]  # short ones translate best
    public = public[~public["text"].astype(str).map(dedupe_key).isin(already_done)]

    attacks = public[public["label"] == 1]
    benign = public[public["label"] == 0]
    n_attack = min(n // 2, len(attacks))
    n_benign = min(n - n_attack, len(benign))
    seed = len(already_done)  # different rows each time you run it
    return pd.concat([attacks.sample(n_attack, random_state=seed),
                      benign.sample(n_benign, random_state=seed)]).reset_index(drop=True)


FAIL_LOG = RAW_DIR / "generation_failures.log"


def _parse_items(content: str) -> dict[int, dict]:
    """Pull {"id", "hinglish", "hindi"} objects out of the model's reply.

    First try the whole JSON array. If that's broken (e.g. the reply was cut off,
    or one translation has a stray quote), fall back to parsing each {...} object
    on its own, so one bad item doesn't throw away the good ones.
    """
    items = []
    match = re.search(r"\[.*\]", content, re.DOTALL)
    if match:
        try:
            items = json.loads(match.group(0))
        except json.JSONDecodeError:
            items = []
    if not items:
        for obj in re.findall(r"\{[^{}]*\}", content, re.DOTALL):
            try:
                items.append(json.loads(obj))
            except json.JSONDecodeError:
                pass
    out = {}
    for item in items:
        if isinstance(item, dict) and "id" in item:
            try:
                out[int(item["id"])] = item
            except (TypeError, ValueError):
                pass
    return out


def _log_failure(texts: list[str], content: str, finish_reason: str | None) -> None:
    """Save failed replies so you can see WHY: refusal, cut-off reply, broken JSON..."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    with FAIL_LOG.open("a", encoding="utf-8") as f:
        f.write(f"--- finish_reason={finish_reason} | {len(texts)} message(s)\n")
        f.write(f"first message: {texts[0][:150]!r}\n")
        f.write(f"reply: {content[:600]!r}\n\n")


def translate_batch(client: OpenAI, texts: list[str]) -> dict[int, dict]:
    numbered = "\n".join(json.dumps({"id": i, "message": t}, ensure_ascii=False) for i, t in enumerate(texts))
    for attempt in range(1, 5):
        try:
            resp = client.chat.completions.create(
                model=os.environ["LLM_MODEL"],
                messages=[{"role": "system", "content": INSTRUCTIONS},
                          {"role": "user", "content": numbered}],
                temperature=0.4,
                # Room for the answer. Reasoning models also "think" before answering,
                # and that thinking uses up tokens too; too small a limit cuts the JSON off.
                max_tokens=6000,
            )
            break
        except RateLimitError:
            print(f"    rate limited, waiting {30 * attempt}s")
            time.sleep(30 * attempt)
    else:
        return {}
    choice = resp.choices[0]
    content = choice.message.content or ""
    results = _parse_items(content)
    if len(results) < len(texts):
        _log_failure(texts, content, getattr(choice, "finish_reason", None))
    return results


def make_rows(src, item) -> list[dict]:
    # group = the original's group, so a translation never lands in a different split
    group = "row-" + dedupe_key(str(src["text"]))
    rows = []
    for lang, field in [("hinglish", "hinglish"), ("hi", "hindi")]:
        text = str(item.get(field, "")).strip()
        if text:
            rows.append(dict(text=text, label=int(src["label"]), lang=lang,
                             category=src["category"], group=group,
                             original=src["text"], keep=""))
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=100, help="how many English examples to translate")
    args = parser.parse_args()

    existing = pd.read_csv(OUT, encoding="utf-8-sig", dtype={"keep": str}) if OUT.exists() else pd.DataFrame()
    done = set(existing["original"].map(dedupe_key)) if not existing.empty else set()

    picked = pick_examples(args.n, done)
    client = OpenAI(base_url=os.environ["LLM_BASE_URL"], api_key=os.environ["LLM_API_KEY"])

    rows, skipped, rescued = [], [], 0
    batches = [picked.iloc[i:i + BATCH_SIZE] for i in range(0, len(picked), BATCH_SIZE)]
    for b, batch in enumerate(batches, 1):
        results = translate_batch(client, batch["text"].astype(str).tolist())
        missing = [i for i in range(len(batch)) if i not in results]
        print(f"  batch {b}/{len(batches)}: {len(batch) - len(missing)}/{len(batch)} translated", end="")

        # Retry each missing example ON ITS OWN. If the model refused the whole batch
        # because of one jailbreak, the other 7 now get through, and we learn exactly
        # which example it refuses.
        for i in missing:
            time.sleep(3)
            single = translate_batch(client, [str(batch.iloc[i]["text"])])
            if 0 in single:
                results[i] = single[0]
                rescued += 1
        print(f", after retry {len(results)}/{len(batch)}")

        for i, (_, src) in enumerate(batch.iterrows()):
            if i in results:
                rows += make_rows(src, results[i])
            else:
                skipped.append(src)
        if b < len(batches):
            time.sleep(PAUSE_SECONDS)

    new = pd.DataFrame(rows)
    out = pd.concat([existing, new], ignore_index=True)
    CUSTOM_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT, index=False, encoding="utf-8-sig")   # utf-8-sig: Excel shows Hindi correctly

    print(f"\nAdded {len(new)} rows. Rescued by single retries: {rescued}.")
    if skipped:
        sk = pd.DataFrame(skipped)
        print(f"Still skipped: {len(sk)} examples "
              f"({(sk['label'] == 1).sum()} attacks, {(sk['label'] == 0).sum()} benign). "
              f"Reasons are in {FAIL_LOG}")
    # Label mix of everything translated so far. If attacks fall far behind benign, the
    # model mostly refused attacks: write more Hinglish attacks yourself in my_examples.csv.
    mix = out.drop_duplicates("original")["label"].value_counts()
    print(f"Translated so far: {mix.get(0, 0)} benign, {mix.get(1, 0)} attacks.")
    print(f"Review them in {OUT} - type y or n in the 'keep' column.")


if __name__ == "__main__":
    main()
