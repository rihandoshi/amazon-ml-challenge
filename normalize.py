"""Vectorized text normalization for business names and addresses.

Everything here is pure-Python / regex / pandas — no external services,
no network calls, no geocoding APIs. Safe under the challenge's fair-play rules.
"""
import re
import unicodedata
import numpy as np
import pandas as pd
from functools import partial
from multiprocessing import Pool

from config import (
    NAME_TOKEN_MAP, ADDRESS_TOKEN_MAP, US_STATES, US_STATE_ABBRS,
    INDIAN_STATES, NUM_WORKERS,
)

_DEVANAGARI_RE = re.compile(r"[\u0900-\u097F]")
_NON_ALNUM_RE = re.compile(r"[^a-z0-9\s]")
_MULTISPACE_RE = re.compile(r"\s+")
_DIGIT_RE = re.compile(r"\d+")
_ZIP5_RE = re.compile(r"\b(\d{5})(-\d{4})?\b")
_PIN6_RE = re.compile(r"\b(\d{6})\b")
# French postal codes are also 5 digits; generic 5-digit pattern for non-US countries
_GENERIC_POSTAL_RE = re.compile(r"\b(\d{5})\b")

# Pattern for house/building numbers at the start of an address
_HOUSE_NO_RE = re.compile(r"^\s*(\d+[a-z]?(?:[\s/-]\d+)?)\b", re.IGNORECASE)

# Regex to detect ANY non-Latin/non-ASCII alphabetic characters
# (used to decide if we need transliteration beyond Devanagari)
# Note: Python's re doesn't support \p{P}, so we list common punct explicitly
_NON_LATIN_ALPHA_RE = re.compile(
    r"[^\u0000-\u024F\u0900-\u097F0-9\s!\"#$%&'()*+,\-./:;<=>?@\[\]^_`{|}~]",
)

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


def transliterate_generic(text: str) -> str:
    """Generic fallback transliteration for ANY non-Latin script.

    Uses Unicode NFKD decomposition to strip diacritics + accents, then
    drops remaining non-ASCII characters.  This handles:
      - Accented Latin (French, German, Spanish, Portuguese)
      - Cyrillic (partial — 'и' → dropped, but combined with embedding it helps)
      - Arabic, Thai, CJK (characters dropped, but embedding model handles them)

    The goal is NOT perfect transliteration — it's to give the fuzzy string
    features *something* to compare rather than empty strings.  The multilingual
    embedding model handles the heavy lifting for non-Latin scripts.
    """
    if not text:
        return text
    # First apply Devanagari-specific transliteration (better quality than generic)
    text = transliterate_devanagari(text)
    # NFKD decomposition separates base characters from combining marks
    nfkd = unicodedata.normalize("NFKD", text)
    # Keep only ASCII characters (base letters after decomposition)
    ascii_text = nfkd.encode("ascii", errors="ignore").decode("ascii")
    return ascii_text


def has_devanagari(text: str) -> bool:
    return bool(text) and bool(_DEVANAGARI_RE.search(text))


def has_non_latin(text: str) -> bool:
    """Check if text contains characters outside Latin/Devanagari scripts."""
    if not text:
        return False
    for ch in text:
        cp = ord(ch)
        # Skip ASCII, Extended Latin, Devanagari, whitespace, digits, punctuation
        if cp <= 0x024F or (0x0900 <= cp <= 0x097F) or ch.isspace() or ch.isdigit():
            continue
        if unicodedata.category(ch).startswith("P"):
            continue
        return True
    return False


def _clean_basic(text: str) -> str:
    if not isinstance(text, str) or not text:
        return ""
    text = unicodedata.normalize("NFKC", text)
    # Apply transliteration: Devanagari first (high quality), then generic fallback
    text = transliterate_generic(text)
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
    """Extract state from address text.

    Language-agnostic: for unknown countries, returns None gracefully.
    """
    if country == "US":
        for full, abbr in US_STATES.items():
            if re.search(rf"\b{re.escape(full)}\b", text):
                return abbr
        for abbr in US_STATE_ABBRS:
            if re.search(rf"\b{abbr.lower()}\b", text):
                return abbr
        return None
    if country == "India":
        # Try longer state names first (e.g. "madhya pradesh" before "pradesh")
        for st in sorted(INDIAN_STATES, key=len, reverse=True):
            if re.search(rf"\b{re.escape(st)}\b", text):
                return st
        return None
    # For unknown countries, don't attempt state extraction
    return None


def _is_likely_city_segment(seg: str, state: str, country: str) -> bool:
    """Determine if a comma-separated segment is likely a city name
    rather than a street address, zip code, or state.

    Heuristics:
    - Not a pure number / zip / pin code
    - Not the state itself
    - Not dominated by house-number-like patterns (e.g. "19 1/2 STARDUST TRAIL")
    - Primarily alphabetic (cities are word-like, not number-heavy)
    """
    seg_cleaned = _clean_basic(seg)
    if not seg_cleaned:
        return False

    # Skip if it matches the state
    if state and state.lower() in seg_cleaned:
        return False

    # Skip pure zip/pin codes
    if _ZIP5_RE.fullmatch(seg.strip()) or _PIN6_RE.fullmatch(seg.strip()):
        return False

    # Skip if it starts with a number (likely a street address)
    if re.match(r"^\d", seg_cleaned):
        return False

    # Skip if the segment is very long and has many digits (likely a full address)
    digit_count = sum(1 for c in seg_cleaned if c.isdigit())
    alpha_count = sum(1 for c in seg_cleaned if c.isalpha())
    if alpha_count == 0:
        return False

    # A city segment should be primarily alphabetic
    # Allow some digits (e.g. in compound names) but not mostly digits
    if digit_count > alpha_count:
        return False

    return True


def _extract_city_improved(raw_addr: str, state: str, zip_code: str, country: str) -> str:
    """Improved city extraction from comma-separated address.

    Handles problematic formats like:
        "GREENSBORO, NC, 19 1/2 STARDUST TRAIL" → "greensboro"
        "1795 Westchester Drive, High Point, NC" → "high point"
        "G-3/571, GULMOHAR COLONY, BHOPAL, Madhya Pradesh" → "bhopal"

    Strategy:
    1. Split on commas
    2. For each segment, check if it looks like a city (alphabetic, not state/zip,
       not a street address)
    3. Among city candidates, prefer:
       - The segment right before the state (if state is present)
       - Otherwise the *first* city-like segment that isn't the first segment
         (first segment is usually a street address)
       - Fallback: the first non-address segment
    """
    segments = [s.strip() for s in re.split(r",", raw_addr) if s.strip()]
    if not segments:
        return None

    # Find all city-candidate segments with their positions
    city_candidates = []
    state_position = -1
    for idx, seg in enumerate(segments):
        seg_l = seg.lower().strip()
        # Track state position
        if state:
            if state.lower() in seg_l:
                state_position = idx
                continue

        if _is_likely_city_segment(seg, state, country):
            city_candidates.append((idx, _clean_basic(seg)))

    if not city_candidates:
        return None

    # Strategy 1: city just before the state position
    if state_position > 0:
        for idx, city in city_candidates:
            if idx == state_position - 1:
                return city

    # Strategy 2: if state is at position 1 (e.g. "GREENSBORO, NC, ..."),
    # the segment at position 0 is the city
    if state_position == 1 and city_candidates:
        # Check if the first segment is actually a city (not a street)
        if city_candidates[0][0] == 0:
            return city_candidates[0][1]

    # Strategy 3: for Indian addresses, city is typically the segment
    # just before the state (already covered) or the last city-like segment
    # before the state
    if state_position >= 0:
        before_state = [(i, c) for i, c in city_candidates if i < state_position]
        if before_state:
            return before_state[-1][1]

    # Strategy 4: for addresses without a recognized state,
    # prefer later city-like segments (last non-address segment tends to be city)
    if len(city_candidates) > 1:
        # Skip the first segment (usually street address), take the last candidate
        non_first = [(i, c) for i, c in city_candidates if i > 0]
        if non_first:
            return non_first[-1][1]

    # Fallback: first candidate
    return city_candidates[0][1]


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

    # --- ZIP/PIN extraction ---
    zip_code = None
    if country == "US":
        m = _ZIP5_RE.search(raw_addr)
        if m:
            zip_code = m.group(1)
    elif country == "India":
        m6 = _PIN6_RE.search(raw_addr)
        if m6:
            zip_code = m6.group(1)
    else:
        # Generic postal code extraction for unknown countries (e.g. France: 5-digit)
        m = _GENERIC_POSTAL_RE.search(raw_addr)
        if m:
            zip_code = m.group(1)

    # --- State extraction ---
    state = _extract_state(norm_addr, country)

    # --- City extraction (improved) ---
    city = _extract_city_improved(raw_addr, state, zip_code, country)

    # --- House number extraction ---
    house_no = None
    lead = _HOUSE_NO_RE.match(cleaned)
    if lead:
        house_no = lead.group(1)

    return {"norm_addr": norm_addr, "zip": zip_code, "state": state, "city": city, "house_no": house_no}


def _normalize_chunk(df_chunk: pd.DataFrame) -> pd.DataFrame:
    """Normalize a single chunk of the dataframe (used for parallel processing)."""
    df_chunk = df_chunk.copy()
    df_chunk["business_name"] = df_chunk["business_name"].fillna("")
    df_chunk["business_address"] = df_chunk["business_address"].fillna("")
    df_chunk["country"] = df_chunk["country"].fillna("UNK")

    name_info = df_chunk["business_name"].apply(normalize_name)
    df_chunk["norm_name"] = name_info.apply(lambda d: d["norm_name"])
    df_chunk["name_tokens"] = name_info.apply(lambda d: d["tokens"])

    addr_info = df_chunk.apply(
        lambda r: normalize_address(r["business_address"], r["country"]), axis=1
    )
    df_chunk["norm_addr"] = addr_info.apply(lambda d: d["norm_addr"])
    df_chunk["zip"] = addr_info.apply(lambda d: d["zip"])
    df_chunk["state"] = addr_info.apply(lambda d: d["state"])
    df_chunk["city"] = addr_info.apply(lambda d: d["city"])
    df_chunk["house_no"] = addr_info.apply(lambda d: d["house_no"])
    df_chunk["has_devanagari"] = df_chunk["business_name"].apply(has_devanagari)
    return df_chunk


def normalize_dataframe(df: pd.DataFrame, n_workers: int = None) -> pd.DataFrame:
    """Apply normalization to a raw source dataframe with
    entity_id, business_name, business_address, country columns.

    Uses multiprocessing to parallelize across CPU cores for large datasets.
    Falls back to single-process for small dataframes (<1000 rows) or if
    multiprocessing fails (e.g. Windows without __main__ guard).
    """
    if n_workers is None:
        n_workers = NUM_WORKERS

    # For small datasets, single-process is faster (avoids serialization overhead)
    if len(df) < 1000 or n_workers <= 1:
        return _normalize_chunk(df)

    # Split into chunks and process in parallel
    chunks = np.array_split(df, n_workers)
    # Filter out empty chunks
    chunks = [c for c in chunks if len(c) > 0]

    print(f"    Normalizing {len(df):,} rows across {len(chunks)} workers...")
    try:
        with Pool(processes=len(chunks)) as pool:
            results = pool.map(_normalize_chunk, chunks)
        return pd.concat(results, ignore_index=True)
    except (RuntimeError, OSError) as e:
        # Fallback: single-process (happens on Windows without __main__ guard)
        print(f"    Multiprocessing failed ({e.__class__.__name__}), falling back to single-process")
        return _normalize_chunk(df)
