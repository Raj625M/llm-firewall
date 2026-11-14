"""
Small helpers shared by all the Day 2 data scripts.

Every example in our dataset has the same columns:
  text     - the prompt or document text
  label    - 0 = benign (safe), 1 = malicious (injection / jailbreak)
  lang     - en / hinglish / hi / other
  category - what kind of example it is (see CATEGORIES below)
  source   - where it came from (dataset name or file name)
  group    - examples with the same group always land in the same split
             (train / val / test). This prevents "leakage": e.g. an English
             attack in train and its Hinglish translation in test would make
             the test score look better than it really is.
"""

import re
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent
RAW_DIR = DATA_DIR / "raw"            # downloaded public data (not committed to git)
CUSTOM_DIR = DATA_DIR / "custom"      # hand-written + reviewed examples (committed)
PROCESSED_DIR = DATA_DIR / "processed"  # final train / val / test splits

COLUMNS = ["text", "label", "lang", "category", "source", "group"]

CATEGORIES = {
    "benign",              # normal user message
    "hard_negative",       # safe, but LOOKS suspicious ("how do I ignore warnings in python?")
    "injection",           # tries to override the app's instructions
    "jailbreak",           # role-play / framing to get forbidden output
    "indirect_injection",  # attack hidden inside a document, email, review...
    "benign_document",     # a normal document with no attack (pairs with indirect_injection)
}

DEVANAGARI = re.compile(r"[ऀ-ॿ]")


def guess_lang(text: str) -> str:
    """Very rough language tag for public data.

    - Any Devanagari letters -> 'hi'
    - A few very common German words -> 'other' (one public dataset mixes in German)
    - Otherwise 'en'
    Our own Hinglish files set lang explicitly, so we never have to guess Hinglish.
    """
    if DEVANAGARI.search(text):
        return "hi"
    lowered = f" {text.lower()} "
    german_hits = sum(w in lowered for w in [" und ", " der ", " die ", " ist ", " nicht ", " ich ", " sie "])
    if german_hits >= 2 or any(ch in text for ch in "äöüß"):
        return "other"
    return "en"


def dedupe_key(text: str) -> str:
    """Normalise text so trivial differences don't count as different examples.

    "Ignore all  previous instructions!!" and "ignore all previous instructions"
    get the same key. \\w keeps letters/digits in any script, so Hindi survives.
    """
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)   # drop punctuation
    return re.sub(r"\s+", " ", text).strip()
