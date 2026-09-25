"""Vectorized text normalization for business names and addresses.

Everything here is pure-Python / regex / pandas — no external services,
no network calls, no geocoding APIs. Safe under the challenge's fair-play rules.
"""
import re
import unicodedata
import pandas as pd

from config import NAME_TOKEN_MAP, ADDRESS_TOKEN_MAP, US_STATES, US_STATE_ABBRS, INDIAN_STATES

_DEVANAGARI_RE = re.compile(r"[\u0900-\u097F]")
_NON_ALNUM_RE = re.compile(r"[^a-z0-9\s]")
_MULTISPACE_RE = re.compile(r"\s+")
_DIGIT_RE = re.compile(r"\d+")
_ZIP5_RE = re.compile(r"\b(\d{5})(-\d{4})?\b")
_PIN6_RE = re.compile(r"\b(\d{6})\b")

# Minimal Devanagari -> Latin phonetic transliteration table.
# This is a *rule-based character map*, not a lookup service — same category as
# using a stemmer. It won't be perfect but gives the embedding model and the
# fuzzy-matching features a consistent script to compare against S1 (always Latin).
_DEVANAGARI_MAP = {
    "अ": "a", "आ": "aa", "इ": "i", "ई": "ii", "उ": "u", "ऊ": "uu", "ऋ": "ri",
    "ए": "e", "ऐ": "ai", "ओ": "o", "औ": "au",
    "क": "k", "ख": "kh", "ग": "g", "घ": "gh", "ङ": "ng",
    "च": "ch", "छ": "chh", "ज": "j", "झ": "jh", "ञ": "ny",
    "ट": "t", "ठ": "th", "ड": "d", "ढ": "dh", "ण": "n",
    "त": "t", "थ": "th", "द": "d", "ध": "dh", "न": "n",
    "प": "p", "फ": "ph", "ब": "b", "भ": "bh", "म": "m",
    "य": "y", "र": "r", "ल": "l", "व": "v",
    "श": "sh", "ष": "sh", "स": "s", "ह": "h",
    "क्ष": "ksh", "त्र": "tr", "ज्ञ": "gy",
    "ा": "a", "ि": "i", "ी": "i", "ु": "u", "ू": "u", "े": "e", "ै": "ai",
    "ो": "o", "ौ": "au", "ं": "n", "ः": "h", "ँ": "n", "्": "",
    "०": "0", "१": "1", "२": "2", "३": "3", "४": "4",
    "५": "5", "६": "6", "७": "7", "८": "8", "९": "9",
}


def transliterate_devanagari(text: str) -> str:
    """Crude but deterministic Devanagari -> Latin phonetic mapping."""
    if not text or not _DEVANAGARI_RE.search(text):
        return text
    out = []
    for ch in text:
        out.append(_DEVANAGARI_MAP.get(ch, ch))
    return "".join(out)


def has_devanagari(text: str) -> bool:
    return bool(text) and bool(_DEVANAGARI_RE.search(text))


def _clean_basic(text: str) -> str:
    if not isinstance(text, str) or not text:
        return ""
    text = unicodedata.normalize("NFKC", text)
    text = transliterate_devanagari(text)
    text = text.lower()
    text = _NON_ALNUM_RE.sub(" ", text)
    text = _MULTISPACE_RE.sub(" ", text).strip()
    return text


def normalize_name(raw_name: str) -> dict:
    """Returns normalized name plus a stable, deduped token list.

    - expands legal-suffix abbreviations to a canonical form
    - drops leading 'the'
    - keeps original token order except leading article
    """
    cleaned = _clean_basic(raw_name)
    if not cleaned:
        return {"norm_name": "", "tokens": []}

    tokens = cleaned.split(" ")
    mapped = []
    for tok in tokens:
        repl = NAME_TOKEN_MAP.get(tok, tok)
        if repl:
            mapped.append(repl)
    norm_name = " ".join(mapped).strip()
    return {"norm_name": norm_name, "tokens": mapped}


def _extract_state(text: str, country: str):
    if country == "US":
        for full, abbr in US_STATES.items():
            if re.search(rf"\b{re.escape(full)}\b", text):
                return abbr
        for abbr in US_STATE_ABBRS:
            if re.search(rf"\b{abbr.lower()}\b", text):
                return abbr
        return None
    if country == "India":
        for st in INDIAN_STATES:
            if re.search(rf"\b{re.escape(st)}\b", text):
                return st
        return None
    return None


def normalize_address(raw_addr: str, country: str) -> dict:
    """Extracts a normalized address string plus coarse structured fields:
    zip/pin code, state, and a best-guess city token, used both as
    blocking keys and as pairwise features.
    """
    cleaned = _clean_basic(raw_addr)
    if not cleaned:
        return {"norm_addr": "", "zip": None, "state": None, "city": None, "house_no": None}

    tokens = cleaned.split(" ")
    mapped = [ADDRESS_TOKEN_MAP.get(t, t) for t in tokens]
    norm_addr = " ".join(mapped)

    zip_code = None
    m = _ZIP5_RE.search(raw_addr) if country == "US" else None
    if m:
        zip_code = m.group(1)
    m6 = _PIN6_RE.search(raw_addr) if country == "India" else None
    if m6:
        zip_code = m6.group(1)

    state = _extract_state(norm_addr, country)

    # crude city guess: comma-separated segment right before the state/zip,
    # falling back to the second-to-last comma segment.
    segments = [s.strip() for s in re.split(r",", raw_addr) if s.strip()]
    city = None
    if segments:
        for seg in reversed(segments):
            seg_l = seg.lower()
            if state and state.lower() in seg_l:
                continue
            if _ZIP5_RE.fullmatch(seg.strip()) or _PIN6_RE.fullmatch(seg.strip()):
                continue
            city = _clean_basic(seg)
            if city:
                break

    house_no = None
    m = _DIGIT_RE.search(mapped[0]) if mapped else None
    # first standalone leading number in the address, common for street numbers
    lead = re.match(r"^\s*(\d+[a-z]?)", cleaned)
    if lead:
        house_no = lead.group(1)

    return {"norm_addr": norm_addr, "zip": zip_code, "state": state, "city": city, "house_no": house_no}


def normalize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Apply normalization to a raw source dataframe with
    entity_id, business_name, business_address, country columns.
    """
    df = df.copy()
    df["business_name"] = df["business_name"].fillna("")
    df["business_address"] = df["business_address"].fillna("")
    df["country"] = df["country"].fillna("UNK")

    name_info = df["business_name"].apply(normalize_name)
    df["norm_name"] = name_info.apply(lambda d: d["norm_name"])
    df["name_tokens"] = name_info.apply(lambda d: d["tokens"])

    addr_info = df.apply(lambda r: normalize_address(r["business_address"], r["country"]), axis=1)
    df["norm_addr"] = addr_info.apply(lambda d: d["norm_addr"])
    df["zip"] = addr_info.apply(lambda d: d["zip"])
    df["state"] = addr_info.apply(lambda d: d["state"])
    df["city"] = addr_info.apply(lambda d: d["city"])
    df["house_no"] = addr_info.apply(lambda d: d["house_no"])
    df["has_devanagari"] = df["business_name"].apply(has_devanagari)
    return df
