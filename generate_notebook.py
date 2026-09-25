"""Generate the complete SageMaker Jupyter notebook for the entity-resolution pipeline.

Run:  .venv\\Scripts\\python.exe generate_notebook.py
Creates: sagemaker_pipeline.ipynb
"""
import json, os

def md(source):
    """Create a markdown cell."""
    return {"cell_type": "markdown", "metadata": {}, "source": [source]}

def code(source):
    """Create a code cell."""
    return {"cell_type": "code", "metadata": {}, "source": [source],
            "outputs": [], "execution_count": None}

cells = []

# =========================================================================
# CELL 1: Title
# =========================================================================
cells.append(md(r"""# Business Entity Resolution Pipeline — Amazon ML Challenge

**Architecture:**
```
S1 records → normalization → lexical/token blocking + FAISS ANN
  → fair candidate ranking (separate lexical + ANN pools)
  → pairwise features (fuzzy + char n-gram + ANN score)
  → LightGBM → threshold tuned for macro F0.5 → final matches
```

**Key improvements in this notebook:**
1. **Frequency-aware token blocking** — prevents memory explosion on large data
2. **Fair candidate capping** — ANN candidates get guaranteed slots
3. **Candidate recall diagnostics** — measures where true matches are lost
4. **Improved address parsing** — handles tricky comma-separated formats
5. **Character n-gram features** — captures typos, abbreviations, transliterations
6. **Correct E5 query/passage prefixes** for ANN retrieval
7. **Ablation flags** — easily toggle each component on/off

**Run each cell in order.** Markdown cells explain what each section does.
"""))

# =========================================================================
# CELL 2: Install dependencies
# =========================================================================
cells.append(md("## 1. Install Dependencies"))
cells.append(code("""!pip install -q pandas>=2.1 numpy>=1.26 rapidfuzz>=3.6 lightgbm>=4.3 scikit-learn>=1.4 pyarrow>=15.0
!pip install -q sentence-transformers>=2.6 faiss-cpu>=1.8 torch>=2.2

import sys
print(f"Python: {sys.version}")
print("All dependencies installed.")
"""))

# =========================================================================
# CELL 3: Configuration
# =========================================================================
cells.append(md("""## 2. Configuration

**Edit these paths and flags before running the pipeline.**

### Data paths
- Upload your dataset to SageMaker (see the README guide)
- Set `DATA_DIR` to the folder containing your `.tsv` files

### Experiment flags
- Toggle features on/off for ablation experiments
- Adjust blocking thresholds and candidate caps
"""))

cells.append(code("""import os
import re
import time
import json
import unicodedata
from collections import defaultdict

import numpy as np
import pandas as pd
from rapidfuzz import fuzz, distance

# =====================================================================
# DATA PATHS — EDIT THESE
# =====================================================================
# For SageMaker Notebook Instance:
#   DATA_DIR = "/home/ec2-user/SageMaker/dataset/train"
#   TEST_DIR = "/home/ec2-user/SageMaker/dataset/test"
# For SageMaker Studio Lab:
#   DATA_DIR = "/home/studio-lab-user/dataset/train"
#   TEST_DIR = "/home/studio-lab-user/dataset/test"
# For SageMaker Studio:
#   DATA_DIR = "/home/sagemaker-user/dataset/train"
#   TEST_DIR = "/home/sagemaker-user/dataset/test"

DATA_DIR = "/home/ec2-user/SageMaker/dataset/train"
TEST_DIR = "/home/ec2-user/SageMaker/dataset/test"
OUT_DIR  = "/home/ec2-user/SageMaker/artifacts"
os.makedirs(OUT_DIR, exist_ok=True)

# =====================================================================
# EXPERIMENT FLAGS — toggle for ablation
# =====================================================================
USE_TOKEN_BLOCKING          = True
USE_ANN                     = True    # set False for a faster token-only run
USE_SEPARATE_NAME_ANN       = False   # experimental: ANN on name-only text
USE_ADDRESS_ANN             = False   # experimental: ANN on address-only text
USE_CHAR_NGRAM_FEATURES     = True
USE_ASYMMETRIC_E5_PREFIXES  = True    # "query:" for S1, "passage:" for S2/S3

# =====================================================================
# BLOCKING SAFETY LIMITS
# =====================================================================
MAX_TOKEN_BLOCK_PAIRS = 5_000   # skip (country,token) blocks with est. pairs > this
MAX_TOKEN_FREQ        = 2_000   # skip tokens appearing in > this many records on either side

# =====================================================================
# CANDIDATE CAPS (per entity, per pool)
# =====================================================================
CANDIDATE_CAP_LEXICAL = 40     # guaranteed slots for lexical candidates
CANDIDATE_CAP_ANN     = 20     # guaranteed slots for ANN candidates

# =====================================================================
# ANN RETRIEVAL
# =====================================================================
ANN_TOP_K     = 10             # neighbours per query
ANN_BATCH     = 512            # embedding batch size
MODEL_ID_EMBEDDING = "intfloat/multilingual-e5-small"

# =====================================================================
# TRAINING
# =====================================================================
MAX_NEGATIVES_PER_ENTITY = 20
VAL_FRAC   = 0.15
RANDOM_SEED = 42

print("Configuration loaded.")
print(f"  DATA_DIR = {DATA_DIR}")
print(f"  TEST_DIR = {TEST_DIR}")
print(f"  OUT_DIR  = {OUT_DIR}")
"""))

# =========================================================================
# CELL 4: Constants (name/address maps, gazetteers)
# =========================================================================
cells.append(md("## 3. Constants — Name/Address Maps & Gazetteers"))
cells.append(code('''# Legal-suffix / common-word normalization for business names.
NAME_TOKEN_MAP = {
    "corp": "corporation", "corporation": "corporation",
    "inc": "incorporated", "incorporated": "incorporated",
    "ltd": "limited", "limited": "limited",
    "llc": "llc", "llp": "llp",
    "pvt": "private", "private": "private",
    "co": "company", "company": "company", "cos": "company",
    "assoc": "associates", "associates": "associates",
    "grp": "group", "group": "group",
    "intl": "international", "international": "international",
    "svcs": "services", "svc": "service", "services": "services", "service": "service",
    "mfg": "manufacturing", "manufacturing": "manufacturing",
    "bros": "brothers", "brothers": "brothers",
    "and": "and", "&": "and",
    "the": "",
}

NAME_STOPWORDS = {
    "the", "and", "of", "&", "a", "an", "for", "llc", "llp", "inc", "incorporated",
    "corp", "corporation", "ltd", "limited", "pvt", "private", "co", "company",
    "group", "services", "service", "international", "intl",
}

ADDRESS_TOKEN_MAP = {
    "rd": "road", "road": "road", "st": "street", "str": "street", "street": "street",
    "ave": "avenue", "av": "avenue", "avenue": "avenue",
    "blvd": "boulevard", "boulevard": "boulevard",
    "dr": "drive", "drive": "drive", "ln": "lane", "lane": "lane",
    "ct": "court", "court": "court", "pl": "place", "place": "place",
    "sq": "square", "square": "square",
    "apt": "apartment", "apartment": "apartment",
    "bldg": "building", "building": "building",
    "flr": "floor", "floor": "floor", "unit": "unit",
    "hwy": "highway", "highway": "highway",
    "pkwy": "parkway", "parkway": "parkway",
    "no": "number", "number": "number", "po": "postoffice",
    "near": "near", "opp": "opposite", "opposite": "opposite",
}

US_STATES = {
    "alabama": "AL", "alaska": "AK", "arizona": "AZ", "arkansas": "AR", "california": "CA",
    "colorado": "CO", "connecticut": "CT", "delaware": "DE", "florida": "FL", "georgia": "GA",
    "hawaii": "HI", "idaho": "ID", "illinois": "IL", "indiana": "IN", "iowa": "IA",
    "kansas": "KS", "kentucky": "KY", "louisiana": "LA", "maine": "ME", "maryland": "MD",
    "massachusetts": "MA", "michigan": "MI", "minnesota": "MN", "mississippi": "MS",
    "missouri": "MO", "montana": "MT", "nebraska": "NE", "nevada": "NV",
    "new hampshire": "NH", "new jersey": "NJ", "new mexico": "NM", "new york": "NY",
    "north carolina": "NC", "north dakota": "ND", "ohio": "OH", "oklahoma": "OK",
    "oregon": "OR", "pennsylvania": "PA", "rhode island": "RI", "south carolina": "SC",
    "south dakota": "SD", "tennessee": "TN", "texas": "TX", "utah": "UT", "vermont": "VT",
    "virginia": "VA", "washington": "WA", "west virginia": "WV", "wisconsin": "WI",
    "wyoming": "WY",
}
US_STATE_ABBRS = set(US_STATES.values())

INDIAN_STATES = {
    "andhra pradesh", "arunachal pradesh", "assam", "bihar", "chhattisgarh", "goa",
    "gujarat", "haryana", "himachal pradesh", "jharkhand", "karnataka", "kerala",
    "madhya pradesh", "maharashtra", "manipur", "meghalaya", "mizoram", "nagaland",
    "odisha", "punjab", "rajasthan", "sikkim", "tamil nadu", "telangana", "tripura",
    "uttar pradesh", "uttarakhand", "west bengal", "delhi", "new delhi", "west delhi",
    "east delhi", "north delhi", "south delhi", "jammu and kashmir", "ladakh",
    "puducherry", "chandigarh",
}

print("Constants loaded.")
'''))

# =========================================================================
# CELL 5: Normalization functions
# =========================================================================
cells.append(md("""## 4. Normalization Functions

Handles:
- Devanagari → Latin transliteration (rule-based, no external API)
- Business name abbreviation expansion
- Address parsing with **improved city extraction**
- Generic postal code extraction (works for US, India, France, etc.)
"""))

cells.append(code(u'''# --- Regex patterns ---
_DEVANAGARI_RE  = re.compile(r"[\\u0900-\\u097F]")
_NON_ALNUM_RE   = re.compile(r"[^a-z0-9\\s]")
_MULTISPACE_RE  = re.compile(r"\\s+")
_DIGIT_RE       = re.compile(r"\\d+")
_ZIP5_RE        = re.compile(r"\\b(\\d{5})(-\\d{4})?\\b")
_PIN6_RE        = re.compile(r"\\b(\\d{6})\\b")
_GENERIC_POSTAL_RE = re.compile(r"\\b(\\d{5})\\b")
_HOUSE_NO_RE    = re.compile(r"^\\s*(\\d+[a-z]?(?:[\\s/-]\\d+)?)\\b", re.IGNORECASE)

# Minimal Devanagari -> Latin phonetic transliteration table.
_DEVANAGARI_MAP = {
    "\\u0905": "a", "\\u0906": "aa", "\\u0907": "i", "\\u0908": "ii",
    "\\u0909": "u", "\\u090A": "uu", "\\u090B": "ri",
    "\\u090F": "e", "\\u0910": "ai", "\\u0913": "o", "\\u0914": "au",
    "\\u0915": "k", "\\u0916": "kh", "\\u0917": "g", "\\u0918": "gh", "\\u0919": "ng",
    "\\u091A": "ch", "\\u091B": "chh", "\\u091C": "j", "\\u091D": "jh", "\\u091E": "ny",
    "\\u091F": "t", "\\u0920": "th", "\\u0921": "d", "\\u0922": "dh", "\\u0923": "n",
    "\\u0924": "t", "\\u0925": "th", "\\u0926": "d", "\\u0927": "dh", "\\u0928": "n",
    "\\u092A": "p", "\\u092B": "ph", "\\u092C": "b", "\\u092D": "bh", "\\u092E": "m",
    "\\u092F": "y", "\\u0930": "r", "\\u0932": "l", "\\u0935": "v",
    "\\u0936": "sh", "\\u0937": "sh", "\\u0938": "s", "\\u0939": "h",
    "\\u0915\\u094D\\u0937": "ksh", "\\u0924\\u094D\\u0930": "tr", "\\u091C\\u094D\\u091E": "gy",
    "\\u093E": "a", "\\u093F": "i", "\\u0940": "i", "\\u0941": "u", "\\u0942": "u",
    "\\u0947": "e", "\\u0948": "ai", "\\u094B": "o", "\\u094C": "au",
    "\\u0902": "n", "\\u0903": "h", "\\u0901": "n", "\\u094D": "",
    "\\u0966": "0", "\\u0967": "1", "\\u0968": "2", "\\u0969": "3", "\\u096A": "4",
    "\\u096B": "5", "\\u096C": "6", "\\u096D": "7", "\\u096E": "8", "\\u096F": "9",
}


def transliterate_devanagari(text):
    if not text or not _DEVANAGARI_RE.search(text):
        return text
    return "".join(_DEVANAGARI_MAP.get(ch, ch) for ch in text)


def has_devanagari(text):
    return bool(text) and bool(_DEVANAGARI_RE.search(text))


def _clean_basic(text):
    if not isinstance(text, str) or not text:
        return ""
    text = unicodedata.normalize("NFKC", text)
    text = transliterate_devanagari(text)
    text = text.lower()
    text = _NON_ALNUM_RE.sub(" ", text)
    text = _MULTISPACE_RE.sub(" ", text).strip()
    return text


def normalize_name(raw_name):
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


def _extract_state(text, country):
    if country == "US":
        for full, abbr in US_STATES.items():
            if re.search(rf"\\b{re.escape(full)}\\b", text):
                return abbr
        for abbr in US_STATE_ABBRS:
            if re.search(rf"\\b{abbr.lower()}\\b", text):
                return abbr
        return None
    if country == "India":
        for st in sorted(INDIAN_STATES, key=len, reverse=True):
            if re.search(rf"\\b{re.escape(st)}\\b", text):
                return st
        return None
    return None


def _is_likely_city_segment(seg, state, country):
    seg_cleaned = _clean_basic(seg)
    if not seg_cleaned:
        return False
    if state and state.lower() in seg_cleaned:
        return False
    if _ZIP5_RE.fullmatch(seg.strip()) or _PIN6_RE.fullmatch(seg.strip()):
        return False
    if re.match(r"^\\d", seg_cleaned):
        return False
    digit_count = sum(1 for c in seg_cleaned if c.isdigit())
    alpha_count = sum(1 for c in seg_cleaned if c.isalpha())
    if alpha_count == 0:
        return False
    if digit_count > alpha_count:
        return False
    return True


def _extract_city_improved(raw_addr, state, zip_code, country):
    """Improved city extraction handling formats like 'GREENSBORO, NC, 19 1/2 STARDUST TRAIL'."""
    segments = [s.strip() for s in re.split(r",", raw_addr) if s.strip()]
    if not segments:
        return None
    city_candidates = []
    state_position = -1
    for idx, seg in enumerate(segments):
        seg_l = seg.lower().strip()
        if state and state.lower() in seg_l:
            state_position = idx
            continue
        if _is_likely_city_segment(seg, state, country):
            city_candidates.append((idx, _clean_basic(seg)))
    if not city_candidates:
        return None
    if state_position > 0:
        for idx, city in city_candidates:
            if idx == state_position - 1:
                return city
    if state_position == 1 and city_candidates:
        if city_candidates[0][0] == 0:
            return city_candidates[0][1]
    if state_position >= 0:
        before_state = [(i, c) for i, c in city_candidates if i < state_position]
        if before_state:
            return before_state[-1][1]
    if len(city_candidates) > 1:
        non_first = [(i, c) for i, c in city_candidates if i > 0]
        if non_first:
            return non_first[-1][1]
    return city_candidates[0][1]


def normalize_address(raw_addr, country):
    cleaned = _clean_basic(raw_addr)
    if not cleaned:
        return {"norm_addr": "", "zip": None, "state": None, "city": None, "house_no": None}
    tokens = cleaned.split(" ")
    mapped = [ADDRESS_TOKEN_MAP.get(t, t) for t in tokens]
    norm_addr = " ".join(mapped)

    zip_code = None
    if country == "US":
        m = _ZIP5_RE.search(raw_addr)
        if m: zip_code = m.group(1)
    elif country == "India":
        m6 = _PIN6_RE.search(raw_addr)
        if m6: zip_code = m6.group(1)
    else:
        m = _GENERIC_POSTAL_RE.search(raw_addr)
        if m: zip_code = m.group(1)

    state = _extract_state(norm_addr, country)
    city = _extract_city_improved(raw_addr, state, zip_code, country)

    house_no = None
    lead = _HOUSE_NO_RE.match(cleaned)
    if lead:
        house_no = lead.group(1)

    return {"norm_addr": norm_addr, "zip": zip_code, "state": state, "city": city, "house_no": house_no}


def normalize_dataframe(df):
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


# Quick sanity check
_test = normalize_address("GREENSBORO, NC, 19 1/2 STARDUST TRAIL", "US")
assert "greensboro" in (_test["city"] or "").lower(), f"City extraction failed: {_test}"
print(f"Address sanity check PASSED: city={_test['city']}, state={_test['state']}")

_test2 = normalize_name("The ABC Corp.")
print(f"Name sanity check: '{_test2['norm_name']}' tokens={_test2['tokens']}")
print("Normalization functions loaded.")
'''))

# =========================================================================
# CELL 6: Blocking functions
# =========================================================================
cells.append(md("""## 5. Blocking Functions (Frequency-Aware, Memory-Safe)

**This is the critical fix.** The original token blocking did an unrestricted many-to-many merge on `(country, token)`, which caused a 146 GiB memory allocation error.

**Solution:** Before merging, estimate `s1_count * other_count` per token block and skip blocks that exceed `MAX_TOKEN_BLOCK_PAIRS`.
"""))

cells.append(code("""def _token_frame(df, min_len=3):
    ex = df[["entity_id", "country", "name_tokens"]].explode("name_tokens")
    ex = ex.rename(columns={"name_tokens": "token"})
    ex = ex[ex["token"].notna() & (ex["token"].str.len() >= min_len)]
    ex = ex[~ex["token"].isin(NAME_STOPWORDS)]
    return ex


def _pairs_from_token_join(s1_tok, other_tok,
                            max_block_pairs=None, max_token_freq=None):
    \"\"\"Frequency-aware token blocking: filters oversized blocks BEFORE merge.\"\"\"
    if max_block_pairs is None: max_block_pairs = MAX_TOKEN_BLOCK_PAIRS
    if max_token_freq is None: max_token_freq = MAX_TOKEN_FREQ

    s1_freq = s1_tok.groupby(["country", "token"])["entity_id"].nunique().reset_index(name="s1_count")
    other_freq = other_tok.groupby(["country", "token"])["entity_id"].nunique().reset_index(name="other_count")
    block_stats = s1_freq.merge(other_freq, on=["country", "token"], how="inner")
    block_stats["est_pairs"] = block_stats["s1_count"] * block_stats["other_count"]

    oversized = block_stats[
        (block_stats["est_pairs"] > max_block_pairs) |
        (block_stats["s1_count"] > max_token_freq) |
        (block_stats["other_count"] > max_token_freq)
    ]
    safe = block_stats[
        (block_stats["est_pairs"] <= max_block_pairs) &
        (block_stats["s1_count"] <= max_token_freq) &
        (block_stats["other_count"] <= max_token_freq)
    ]

    n_skipped = len(oversized)
    pairs_avoided = int(oversized["est_pairs"].sum()) if n_skipped > 0 else 0
    safe_est = int(safe["est_pairs"].sum()) if len(safe) > 0 else 0
    print(f"    Blocks: {len(block_stats)} total, {n_skipped} skipped ({pairs_avoided:,} pairs avoided), {len(safe)} safe ({safe_est:,} est. pairs)")

    if safe.empty:
        return pd.DataFrame(columns=["source1_entity_id", "other_entity_id"])

    safe_keys = safe[["country", "token"]]
    s1_safe = s1_tok.merge(safe_keys, on=["country", "token"], how="inner")
    other_safe = other_tok.merge(safe_keys, on=["country", "token"], how="inner")
    merged = s1_safe.merge(other_safe, on=["country", "token"], suffixes=("_s1", "_other"))
    pairs = merged[["entity_id_s1", "entity_id_other"]].drop_duplicates()
    pairs.columns = ["source1_entity_id", "other_entity_id"]
    return pairs


def _pairs_from_zip_join(s1, other):
    a = s1[s1["zip"].notna()][["entity_id", "country", "zip"]]
    b = other[other["zip"].notna()][["entity_id", "country", "zip"]]
    if a.empty or b.empty:
        return pd.DataFrame(columns=["source1_entity_id", "other_entity_id"])
    merged = a.merge(b, on=["country", "zip"], suffixes=("_s1", "_other"))
    pairs = merged[["entity_id_s1", "entity_id_other"]].drop_duplicates()
    pairs.columns = ["source1_entity_id", "other_entity_id"]
    return pairs


def _pairs_from_city_firsttoken_join(s1, other):
    def first_tok(tokens):
        return tokens[0] if isinstance(tokens, list) and tokens else None
    a = s1.assign(first_token=s1["name_tokens"].apply(first_tok))
    b = other.assign(first_token=other["name_tokens"].apply(first_tok))
    a = a[a["city"].notna() & a["first_token"].notna()][["entity_id", "country", "city", "first_token"]]
    b = b[b["city"].notna() & b["first_token"].notna()][["entity_id", "country", "city", "first_token"]]
    if a.empty or b.empty:
        return pd.DataFrame(columns=["source1_entity_id", "other_entity_id"])
    merged = a.merge(b, on=["country", "city", "first_token"], suffixes=("_s1", "_other"))
    pairs = merged[["entity_id_s1", "entity_id_other"]].drop_duplicates()
    pairs.columns = ["source1_entity_id", "other_entity_id"]
    return pairs


def generate_token_candidates(s1_df, other_df, per_country=True):
    all_pairs = []
    countries = sorted(set(s1_df["country"].unique()) | set(other_df["country"].unique())) if per_country else [None]
    for c in countries:
        s1_c = s1_df[s1_df["country"] == c] if c else s1_df
        other_c = other_df[other_df["country"] == c] if c else other_df
        if s1_c.empty or other_c.empty:
            continue
        print(f"  Country={c}: S1={len(s1_c):,} Other={len(other_c):,}")
        s1_tok = _token_frame(s1_c)
        other_tok = _token_frame(other_c)
        all_pairs.append(_pairs_from_token_join(s1_tok, other_tok))
        all_pairs.append(_pairs_from_zip_join(s1_c, other_c))
        all_pairs.append(_pairs_from_city_firsttoken_join(s1_c, other_c))
    if not all_pairs:
        return pd.DataFrame(columns=["source1_entity_id", "other_entity_id"])
    out = pd.concat(all_pairs, ignore_index=True).drop_duplicates()
    print(f"  Total token candidates: {len(out):,}")
    return out


print("Blocking functions loaded.")
"""))

# =========================================================================
# CELL 7: ANN functions
# =========================================================================
cells.append(md("""## 6. ANN Embedding Functions (FAISS)

Uses `intfloat/multilingual-e5-small` (MIT license, ~118M params).
- Handles Latin, Devanagari, French, and other scripts
- Correct `query:` / `passage:` prefixes for E5 models
- Optional separate name-only / address-only retrieval
"""))

cells.append(code("""def _embed_texts(model, texts, prefix, batch_size):
    prefixed = [f"{prefix}{t}" for t in texts]
    return model.encode(prefixed, batch_size=batch_size, show_progress_bar=True,
                         normalize_embeddings=True, convert_to_numpy=True).astype("float32")


def _ann_search_single(s1_c, other_c, model, text_fn, top_k, batch_size):
    import faiss
    query_prefix = "query: " if USE_ASYMMETRIC_E5_PREFIXES else "passage: "
    passage_prefix = "passage: "

    other_text = text_fn(other_c)
    other_emb = _embed_texts(model, other_text, passage_prefix, batch_size)
    index = faiss.IndexFlatIP(other_emb.shape[1])
    index.add(other_emb)

    s1_text = text_fn(s1_c)
    s1_emb = _embed_texts(model, s1_text, query_prefix, batch_size)
    sims, idxs = index.search(s1_emb, min(top_k, len(other_c)))

    s1_ids = s1_c["entity_id"].to_numpy()
    other_ids = other_c["entity_id"].to_numpy()
    results = []
    for i in range(len(s1_ids)):
        for j, sim in zip(idxs[i], sims[i]):
            if j < 0: continue
            results.append((s1_ids[i], other_ids[j], float(sim)))
    return results


def generate_embedding_candidates(s1_df, other_df, top_k=None, batch_size=None):
    from sentence_transformers import SentenceTransformer
    if top_k is None: top_k = ANN_TOP_K
    if batch_size is None: batch_size = ANN_BATCH
    model = SentenceTransformer(MODEL_ID_EMBEDDING)

    def combined_text(df): return (df["norm_name"] + " " + df["norm_addr"]).tolist()
    def name_text(df): return df["norm_name"].tolist()
    def addr_text(df): return df["norm_addr"].tolist()

    retrieval_modes = [("combined", combined_text)]
    if USE_SEPARATE_NAME_ANN: retrieval_modes.append(("name", name_text))
    if USE_ADDRESS_ANN: retrieval_modes.append(("address", addr_text))
    print(f"  ANN modes: {[m[0] for m in retrieval_modes]}, top_k={top_k}")

    all_pairs = []
    for c in sorted(set(s1_df["country"].unique()) | set(other_df["country"].unique())):
        s1_c = s1_df[s1_df["country"] == c]
        other_c = other_df[other_df["country"] == c]
        if s1_c.empty or other_c.empty: continue
        print(f"  ANN country={c}: S1={len(s1_c):,} Other={len(other_c):,}")
        for mode_name, text_fn in retrieval_modes:
            print(f"    mode={mode_name}...")
            results = _ann_search_single(s1_c, other_c, model, text_fn, top_k, batch_size)
            all_pairs.extend(results)

    df = pd.DataFrame(all_pairs, columns=["source1_entity_id", "other_entity_id", "ann_score"])
    if not df.empty:
        df = df.sort_values("ann_score", ascending=False).drop_duplicates(
            subset=["source1_entity_id", "other_entity_id"], keep="first")
    print(f"  Total ANN candidates: {len(df):,}")
    return df


print("ANN functions loaded.")
"""))

# =========================================================================
# CELL 8: Fair candidate capping
# =========================================================================
cells.append(md("""## 7. Fair Candidate Capping

**Problem with old approach:** Union all candidates, rank by name `token_set_ratio`, keep top-50. ANN candidates with low lexical similarity get discarded unfairly.

**New approach:** Separate pools with guaranteed slots:
- Top `CANDIDATE_CAP_LEXICAL` from lexical blocking (ranked by name score)
- Top `CANDIDATE_CAP_ANN` from ANN (ranked by ANN cosine similarity)
- Union → deduplicate
"""))

cells.append(code("""def quick_score(pairs, s1_df, other_df):
    \"\"\"Cheap name-based score for ranking lexical candidates (NOT the final score).\"\"\"
    s1_idx = s1_df.set_index("entity_id")["norm_name"]
    other_idx = other_df.set_index("entity_id")["norm_name"]
    left = pairs["source1_entity_id"].map(s1_idx).fillna("")
    right = pairs["other_entity_id"].map(other_idx).fillna("")
    return pd.Series([fuzz.token_set_ratio(a, b) for a, b in zip(left, right)], index=pairs.index)


def cap_candidates_fair(token_pairs, ann_pairs, score_col_lex,
                         score_col_ann="ann_score", max_lex=None, max_ann=None):
    if max_lex is None: max_lex = CANDIDATE_CAP_LEXICAL
    if max_ann is None: max_ann = CANDIDATE_CAP_ANN
    retained = []

    if not token_pairs.empty and score_col_lex in token_pairs.columns:
        top_lex = (token_pairs.sort_values(score_col_lex, ascending=False)
                   .groupby("source1_entity_id", group_keys=False).head(max_lex))
        top_lex = top_lex.assign(from_token_block=1)
        retained.append(top_lex[["source1_entity_id", "other_entity_id", score_col_lex, "from_token_block"]])
    elif not token_pairs.empty:
        token_pairs = token_pairs.assign(from_token_block=1)
        retained.append(token_pairs[["source1_entity_id", "other_entity_id", "from_token_block"]])

    if not ann_pairs.empty and score_col_ann in ann_pairs.columns:
        top_ann = (ann_pairs.sort_values(score_col_ann, ascending=False)
                   .groupby("source1_entity_id", group_keys=False).head(max_ann))
        top_ann = top_ann.assign(from_ann=1)
        retained.append(top_ann[["source1_entity_id", "other_entity_id", score_col_ann, "from_ann"]])

    if not retained:
        return pd.DataFrame(columns=["source1_entity_id", "other_entity_id"])

    combined = pd.concat(retained, ignore_index=True)
    agg_cols = {k: "max" for k in ["from_token_block", "from_ann", score_col_lex, score_col_ann]
                if k in combined.columns}
    if agg_cols:
        combined = combined.groupby(["source1_entity_id", "other_entity_id"], as_index=False).agg(agg_cols)
    else:
        combined = combined.drop_duplicates(subset=["source1_entity_id", "other_entity_id"])

    for col in ["from_token_block", "from_ann"]:
        if col in combined.columns:
            combined[col] = combined[col].fillna(0).astype(int)

    print(f"  Fair cap: {len(combined):,} candidates (lex<={max_lex}, ann<={max_ann} per entity)")
    return combined


print("Candidate capping functions loaded.")
"""))

# =========================================================================
# CELL 9: Feature engineering
# =========================================================================
cells.append(md("""## 8. Pairwise Feature Engineering

**20 features** including 3 new character n-gram cosine similarity features:
- `name_char_3gram_cosine` — captures typos, abbreviations, transliteration
- `name_char_4gram_cosine` — longer n-grams, less noisy
- `addr_char_3gram_cosine` — address-level sub-word similarity
"""))

cells.append(code("""FEATURE_COLUMNS = [
    "name_jaro_winkler", "name_levenshtein_ratio", "name_token_sort_ratio",
    "name_token_set_ratio", "name_partial_ratio", "name_len_diff",
    "name_common_token_frac", "name_first_token_match",
    "addr_token_sort_ratio", "addr_partial_ratio",
    "city_exact_match", "city_fuzzy_score", "state_exact_match",
    "zip_exact_match", "zip_present_both", "house_no_exact_match",
    "country_match",
    "name_char_3gram_cosine", "name_char_4gram_cosine", "addr_char_3gram_cosine",
]


def _safe(s):
    return s if isinstance(s, str) else ""


def _char_ngrams(text, n):
    if not text or len(text) < n: return {}
    grams = {}
    for i in range(len(text) - n + 1):
        g = text[i:i+n]
        grams[g] = grams.get(g, 0) + 1
    return grams


def _ngram_cosine_sim(a, b, n):
    if not a or not b: return 0.0
    ga, gb = _char_ngrams(a, n), _char_ngrams(b, n)
    if not ga or not gb: return 0.0
    common = set(ga) & set(gb)
    if not common: return 0.0
    dot = sum(ga[k] * gb[k] for k in common)
    na = sum(v*v for v in ga.values()) ** 0.5
    nb = sum(v*v for v in gb.values()) ** 0.5
    return dot / (na * nb) if na and nb else 0.0


def build_pair_features(pairs, s1_df, other_df):
    s1_idx = s1_df.set_index("entity_id")
    other_idx = other_df.set_index("entity_id")

    left = pairs["source1_entity_id"].map(s1_idx["norm_name"])
    right = pairs["other_entity_id"].map(other_idx["norm_name"])
    left_addr = pairs["source1_entity_id"].map(s1_idx["norm_addr"])
    right_addr = pairs["other_entity_id"].map(other_idx["norm_addr"])
    left_tokens = pairs["source1_entity_id"].map(s1_idx["name_tokens"])
    right_tokens = pairs["other_entity_id"].map(other_idx["name_tokens"])
    left_city = pairs["source1_entity_id"].map(s1_idx["city"])
    right_city = pairs["other_entity_id"].map(other_idx["city"])
    left_state = pairs["source1_entity_id"].map(s1_idx["state"])
    right_state = pairs["other_entity_id"].map(other_idx["state"])
    left_zip = pairs["source1_entity_id"].map(s1_idx["zip"])
    right_zip = pairs["other_entity_id"].map(other_idx["zip"])
    left_house = pairs["source1_entity_id"].map(s1_idx["house_no"])
    right_house = pairs["other_entity_id"].map(other_idx["house_no"])
    left_country = pairs["source1_entity_id"].map(s1_idx["country"])
    right_country = pairs["other_entity_id"].map(other_idx["country"])

    n = len(pairs)
    jaro = np.empty(n); lev = np.empty(n); tsort = np.empty(n)
    tset = np.empty(n); partial = np.empty(n); addr_tsort = np.empty(n)
    addr_partial = np.empty(n); common_frac = np.empty(n); first_tok = np.empty(n)
    city_fuzzy = np.empty(n)
    name_3g = np.zeros(n); name_4g = np.zeros(n); addr_3g = np.zeros(n)

    left_l = left.tolist(); right_l = right.tolist()
    left_addr_l = left_addr.tolist(); right_addr_l = right_addr.tolist()
    left_tokens_l = left_tokens.tolist(); right_tokens_l = right_tokens.tolist()
    left_city_l = left_city.tolist(); right_city_l = right_city.tolist()

    for i in range(n):
        a, b = _safe(left_l[i]), _safe(right_l[i])
        jaro[i] = distance.JaroWinkler.similarity(a, b) if a and b else 0.0
        lev[i] = fuzz.ratio(a, b) / 100.0
        tsort[i] = fuzz.token_sort_ratio(a, b) / 100.0
        tset[i] = fuzz.token_set_ratio(a, b) / 100.0
        partial[i] = fuzz.partial_ratio(a, b) / 100.0

        aa, ab = _safe(left_addr_l[i]), _safe(right_addr_l[i])
        addr_tsort[i] = fuzz.token_sort_ratio(aa, ab) / 100.0 if aa and ab else 0.0
        addr_partial[i] = fuzz.partial_ratio(aa, ab) / 100.0 if aa and ab else 0.0

        lt = left_tokens_l[i] if isinstance(left_tokens_l[i], list) else []
        rt = right_tokens_l[i] if isinstance(right_tokens_l[i], list) else []
        if lt and rt:
            common = len(set(lt) & set(rt))
            common_frac[i] = common / max(len(set(lt) | set(rt)), 1)
            first_tok[i] = 1.0 if lt[0] == rt[0] else 0.0
        else:
            common_frac[i] = 0.0; first_tok[i] = 0.0

        lc, rc = left_city_l[i], right_city_l[i]
        city_fuzzy[i] = fuzz.ratio(_safe(lc), _safe(rc)) / 100.0 if lc and rc else 0.0

        if USE_CHAR_NGRAM_FEATURES:
            name_3g[i] = _ngram_cosine_sim(a, b, 3)
            name_4g[i] = _ngram_cosine_sim(a, b, 4)
            addr_3g[i] = _ngram_cosine_sim(aa, ab, 3)

    out = pd.DataFrame({
        "source1_entity_id": pairs["source1_entity_id"].values,
        "other_entity_id": pairs["other_entity_id"].values,
        "name_jaro_winkler": jaro, "name_levenshtein_ratio": lev,
        "name_token_sort_ratio": tsort, "name_token_set_ratio": tset,
        "name_partial_ratio": partial,
        "name_len_diff": np.abs(pd.Series(left_l).str.len().fillna(0).values -
                                 pd.Series(right_l).str.len().fillna(0).values),
        "name_common_token_frac": common_frac, "name_first_token_match": first_tok,
        "addr_token_sort_ratio": addr_tsort, "addr_partial_ratio": addr_partial,
        "city_exact_match": (left_city.fillna("__L") == right_city.fillna("__R")).astype(int).values,
        "city_fuzzy_score": city_fuzzy,
        "state_exact_match": (left_state.fillna("__L") == right_state.fillna("__R")).astype(int).values,
        "zip_exact_match": ((left_zip.notna()) & (right_zip.notna()) & (left_zip == right_zip)).astype(int).values,
        "zip_present_both": ((left_zip.notna()) & (right_zip.notna())).astype(int).values,
        "house_no_exact_match": ((left_house.notna()) & (right_house.notna()) & (left_house == right_house)).astype(int).values,
        "country_match": (left_country.values == right_country.values).astype(int),
        "name_char_3gram_cosine": name_3g, "name_char_4gram_cosine": name_4g,
        "addr_char_3gram_cosine": addr_3g,
    })
    if "ann_score" in pairs.columns: out["ann_score"] = pairs["ann_score"].values
    for col in ["from_token_block", "from_ann"]:
        if col in pairs.columns: out[col] = pairs[col].values
    return out


print("Feature engineering functions loaded.")
print(f"  {len(FEATURE_COLUMNS)} features: {FEATURE_COLUMNS}")
"""))

# =========================================================================
# CELL 10: Pairs builder + Metrics
# =========================================================================
cells.append(md("## 9. Labeling, Metrics & Candidate Recall"))
cells.append(code("""def build_labeled_pairs(candidate_pairs, ground_truth,
                       max_negatives_per_entity=20, random_state=42):
    gt_map = {}
    for sid, ids in zip(ground_truth["source1_entity_id"], ground_truth["matched_entity_ids"]):
        ids = str(ids).strip()
        gt_map[sid] = set(ids.split(",")) if ids else set()

    cp = candidate_pairs.copy()
    cp["label"] = cp.apply(lambda r: 1 if r["other_entity_id"] in gt_map.get(r["source1_entity_id"], set()) else 0, axis=1)

    positives = cp[cp["label"] == 1]
    negatives = cp[cp["label"] == 0]
    neg_parts = []
    for _, g in negatives.groupby("source1_entity_id", sort=False):
        n = min(len(g), max_negatives_per_entity)
        neg_parts.append(g.sample(n=n, random_state=random_state))
    negatives_capped = pd.concat(neg_parts, ignore_index=True) if neg_parts else negatives
    out = pd.concat([positives, negatives_capped], ignore_index=True)
    return out.sample(frac=1.0, random_state=random_state).reset_index(drop=True)


def group_split_entities(entity_ids, val_frac=0.15, random_state=42):
    rng = np.random.default_rng(random_state)
    ids = np.array(sorted(set(entity_ids)))
    rng.shuffle(ids)
    n_val = int(len(ids) * val_frac)
    return set(ids[n_val:]), set(ids[:n_val])


def _parse_ids(s):
    if s is None or isinstance(s, float): return set()
    s = str(s).strip()
    return set(s.split(",")) if s else set()


def macro_f05(predictions, ground_truth, all_source1_ids):
    scores = []
    for sid in all_source1_ids:
        pred = predictions.get(sid, set())
        true = ground_truth.get(sid, set())
        if not isinstance(pred, set): pred = _parse_ids(pred)
        if not isinstance(true, set): true = _parse_ids(true)
        if not true and not pred: scores.append(1.0); continue
        if not pred: scores.append(0.0); continue
        if not true: scores.append(0.0); continue
        tp = len(pred & true)
        p = tp / len(pred)
        r = tp / len(true)
        f05 = (1.25 * p * r) / (0.25 * p + r) if (0.25 * p + r) > 0 else 0.0
        scores.append(f05)
    return sum(scores) / len(scores) if scores else 0.0


def best_threshold_for_f05(scored_pairs, ground_truth, all_source1_ids, thresholds=None):
    if thresholds is None: thresholds = np.arange(0.05, 0.96, 0.02)
    curve = []
    best_t, best_f = None, -1.0
    for t in thresholds:
        kept = scored_pairs[scored_pairs["score"] >= t]
        preds = defaultdict(set)
        for sid, oid in zip(kept["source1_entity_id"], kept["other_entity_id"]):
            preds[sid].add(oid)
        f = macro_f05(preds, ground_truth, all_source1_ids)
        curve.append((float(t), f))
        if f > best_f: best_f, best_t = f, float(t)

    # Extended report
    if best_t is not None:
        kept = scored_pairs[scored_pairs["score"] >= best_t]
        preds = defaultdict(set)
        for sid, oid in zip(kept["source1_entity_id"], kept["other_entity_id"]):
            preds[sid].add(oid)
        total_tp, total_pred, total_true = 0, 0, 0
        for sid in all_source1_ids:
            pred = preds.get(sid, set())
            true = ground_truth.get(sid, set())
            if not isinstance(true, set): true = _parse_ids(true)
            total_tp += len(pred & true)
            total_pred += len(pred)
            total_true += len(true)
        prec = total_tp / total_pred if total_pred > 0 else 0.0
        rec = total_tp / total_true if total_true > 0 else 0.0
        f1 = (2*prec*rec)/(prec+rec) if (prec+rec) > 0 else 0.0
        print(f"  Best threshold: {best_t:.4f}")
        print(f"  Macro F0.5:     {best_f:.4f}")
        print(f"  Precision:      {prec:.4f}")
        print(f"  Recall:         {rec:.4f}")
        print(f"  F1:             {f1:.4f}")
        print(f"  Predicted:      {total_pred}")

    return best_t, best_f, curve


def candidate_recall(candidate_pairs, ground_truth, source1_ids=None):
    cand_set = defaultdict(set)
    for sid, oid in zip(candidate_pairs["source1_entity_id"], candidate_pairs["other_entity_id"]):
        cand_set[sid].add(oid)
    if source1_ids is None: source1_ids = set(ground_truth.keys())
    total, found, ent_match, ent_full = 0, 0, 0, 0
    for sid in source1_ids:
        true = ground_truth.get(sid, set())
        if not isinstance(true, set): true = _parse_ids(true)
        if not true: continue
        ent_match += 1
        f = len(true & cand_set.get(sid, set()))
        total += len(true); found += f
        if f == len(true): ent_full += 1
    recall = found / total if total > 0 else 0.0
    return {"recall": recall, "found": found, "total": total,
            "entities_with_matches": ent_match, "fully_recalled": ent_full,
            "n_candidates": len(candidate_pairs)}


def evaluate_recall_at_k(candidates, score_col, gt_map, k_values=(20, 50, 100, 200, 500), source1_ids=None):
    if source1_ids is not None:
        candidates = candidates[candidates["source1_entity_id"].isin(source1_ids)]
    print(f"  {'K':>6}  {'Candidates':>12}  {'Recall':>8}  {'Found':>8}  {'Total':>8}")
    print(f"  {'---':>6}  {'---':>12}  {'---':>8}  {'---':>8}  {'---':>8}")
    stats_all = candidate_recall(candidates, gt_map, source1_ids)
    print(f"  {'ALL':>6}  {stats_all['n_candidates']:>12,}  {stats_all['recall']:>8.4f}  {stats_all['found']:>8,}  {stats_all['total']:>8,}")
    results = [("ALL", stats_all)]
    for k in sorted(k_values):
        if score_col in candidates.columns:
            capped = candidates.sort_values(score_col, ascending=False).groupby("source1_entity_id", group_keys=False).head(k)
        else:
            capped = candidates.groupby("source1_entity_id", group_keys=False).head(k)
        stats = candidate_recall(capped, gt_map, source1_ids)
        print(f"  {k:>6}  {stats['n_candidates']:>12,}  {stats['recall']:>8.4f}  {stats['found']:>8,}  {stats['total']:>8,}")
        results.append((k, stats))
    return results


print("Labeling, metrics, and candidate recall functions loaded.")
"""))

# =========================================================================
# CELL 11: Load data
# =========================================================================
cells.append(md("""## 10. Load & Normalize Training Data

This is where the pipeline starts. Loading and normalizing the full dataset will take several minutes.
"""))

cells.append(code("""t_start = time.time()

print("Loading training data...")
s1_raw = pd.read_csv(os.path.join(DATA_DIR, "train_source1.tsv"), sep="\\t")
s2_raw = pd.read_csv(os.path.join(DATA_DIR, "train_source2.tsv"), sep="\\t")
s3_raw = pd.read_csv(os.path.join(DATA_DIR, "train_source3.tsv"), sep="\\t")
gt     = pd.read_csv(os.path.join(DATA_DIR, "train_ground_truth.tsv"), sep="\\t")
gt["matched_entity_ids"] = gt["matched_entity_ids"].fillna("")

print(f"  S1: {len(s1_raw):,} rows")
print(f"  S2: {len(s2_raw):,} rows")
print(f"  S3: {len(s3_raw):,} rows")
print(f"  GT: {len(gt):,} entries")

# Build ground-truth lookup
gt_map = {}
for sid, ids in zip(gt["source1_entity_id"], gt["matched_entity_ids"]):
    ids = str(ids).strip()
    gt_map[sid] = set(ids.split(",")) if ids else set()

n_with = sum(1 for v in gt_map.values() if v)
n_singleton = sum(1 for v in gt_map.values() if not v)
total_true = sum(len(v) for v in gt_map.values())
print(f"  Entities with matches: {n_with:,}")
print(f"  Singleton entities:    {n_singleton:,}")
print(f"  Total true match pairs: {total_true:,}")
"""))

cells.append(code("""print("Normalizing S1...")
s1 = normalize_dataframe(s1_raw)
print("Normalizing S2...")
s2 = normalize_dataframe(s2_raw)
print("Normalizing S3...")
s3 = normalize_dataframe(s3_raw)

for name, df in [("S1", s1), ("S2", s2), ("S3", s3)]:
    print(f"  {name} countries: {dict(df['country'].value_counts())}")

print(f"Normalization done in {time.time()-t_start:.0f}s")
"""))

# =========================================================================
# CELL 12: Token blocking
# =========================================================================
cells.append(md("""## 11. Candidate Generation — Token Blocking

Frequency-aware blocking prevents memory explosion on high-frequency tokens.
"""))

cells.append(code("""print("=" * 60)
print("Token blocking S1 x S2...")
t0 = time.time()
token_cand2 = generate_token_candidates(s1, s2) if USE_TOKEN_BLOCKING else pd.DataFrame(columns=["source1_entity_id", "other_entity_id"])
print(f"  S1xS2 token candidates: {len(token_cand2):,} ({time.time()-t0:.1f}s)")

print("\\nToken blocking S1 x S3...")
t0 = time.time()
token_cand3 = generate_token_candidates(s1, s3) if USE_TOKEN_BLOCKING else pd.DataFrame(columns=["source1_entity_id", "other_entity_id"])
print(f"  S1xS3 token candidates: {len(token_cand3):,} ({time.time()-t0:.1f}s)")
"""))

# =========================================================================
# CELL 13: ANN candidates
# =========================================================================
cells.append(md("""## 12. Candidate Generation — ANN Embedding Retrieval

**Set `USE_ANN = False` above to skip this step** (much faster, but lower recall for cross-script and heavy-typo matches).

This step takes 10-30+ minutes depending on dataset size and whether you have a GPU.
"""))

cells.append(code("""ann2 = pd.DataFrame(columns=["source1_entity_id", "other_entity_id", "ann_score"])
ann3 = pd.DataFrame(columns=["source1_entity_id", "other_entity_id", "ann_score"])

if USE_ANN:
    print("ANN S1 x S2...")
    t0 = time.time()
    ann2 = generate_embedding_candidates(s1, s2)
    print(f"  ANN S1xS2: {len(ann2):,} ({time.time()-t0:.1f}s)")

    print("\\nANN S1 x S3...")
    t0 = time.time()
    ann3 = generate_embedding_candidates(s1, s3)
    print(f"  ANN S1xS3: {len(ann3):,} ({time.time()-t0:.1f}s)")
else:
    print("ANN skipped (USE_ANN=False)")
"""))

# =========================================================================
# CELL 14: Fair candidate capping
# =========================================================================
cells.append(md("## 13. Fair Candidate Capping"))
cells.append(code("""print("Scoring lexical candidates...")
if not token_cand2.empty:
    token_cand2 = token_cand2.assign(_q=quick_score(token_cand2, s1, s2))
if not token_cand3.empty:
    token_cand3 = token_cand3.assign(_q=quick_score(token_cand3, s1, s3))

print("Fair capping S2 candidates...")
cand2 = cap_candidates_fair(token_cand2, ann2, score_col_lex="_q")
print("Fair capping S3 candidates...")
cand3 = cap_candidates_fair(token_cand3, ann3, score_col_lex="_q")

# Drop temp column
if "_q" in cand2.columns: cand2 = cand2.drop(columns="_q")
if "_q" in cand3.columns: cand3 = cand3.drop(columns="_q")

print(f"\\nCandidates after capping: S2={len(cand2):,}  S3={len(cand3):,}")
print(f"Total: {len(cand2)+len(cand3):,}")
"""))

# =========================================================================
# CELL 15: Candidate recall evaluation
# =========================================================================
cells.append(md("""## 14. Candidate Recall Evaluation

**Critical diagnostic:** Shows what fraction of true matches survived candidate generation at various K values.

If candidate recall is very low, increasing K or relaxing blocking thresholds will help more than tuning the classifier.
"""))

cells.append(code("""print("Evaluating candidate recall...")
all_cand = pd.concat([cand2, cand3], ignore_index=True)
cr = candidate_recall(all_cand, gt_map)
print(f"\\nOverall candidate recall: {cr['recall']:.4f}")
print(f"Found {cr['found']:,} / {cr['total']:,} true matches")
print(f"Entities fully recalled: {cr['fully_recalled']:,} / {cr['entities_with_matches']:,}")

# Score all candidates for K evaluation
if "ann_score" in all_cand.columns and all_cand["ann_score"].notna().any():
    score_col = "ann_score"
    # Fill missing ann_score for token-only candidates
    if all_cand["ann_score"].isna().any():
        s23 = pd.concat([s2, s3], ignore_index=True)
        mask = all_cand["ann_score"].isna()
        all_cand.loc[mask, "ann_score"] = quick_score(all_cand[mask], s1, s23).values / 100.0
else:
    s23 = pd.concat([s2, s3], ignore_index=True)
    all_cand["_eval_score"] = quick_score(all_cand, s1, s23)
    score_col = "_eval_score"

print("\\nCandidate recall at various K:")
evaluate_recall_at_k(all_cand, score_col, gt_map, k_values=(20, 50, 100, 200, 500))
"""))

# =========================================================================
# CELL 16: Feature engineering
# =========================================================================
cells.append(md("## 15. Feature Engineering"))
cells.append(code("""print("Building features for S2 candidates...")
t0 = time.time()
feat2 = build_pair_features(cand2, s1, s2)
print(f"  S2 features: {len(feat2):,} pairs ({time.time()-t0:.1f}s)")

print("Building features for S3 candidates...")
t0 = time.time()
feat3 = build_pair_features(cand3, s1, s3)
print(f"  S3 features: {len(feat3):,} pairs ({time.time()-t0:.1f}s)")

print(f"Feature columns: {[c for c in FEATURE_COLUMNS if c in feat2.columns]}")
"""))

# =========================================================================
# CELL 17: Labeling
# =========================================================================
cells.append(md("## 16. Label Pairs & Entity-Level Split"))
cells.append(code("""import lightgbm as lgb

print("Labeling pairs (hard negatives from blocking)...")
labeled2 = build_labeled_pairs(feat2, gt, max_negatives_per_entity=MAX_NEGATIVES_PER_ENTITY)
labeled3 = build_labeled_pairs(feat3, gt, max_negatives_per_entity=MAX_NEGATIVES_PER_ENTITY)
labeled = pd.concat([labeled2, labeled3], ignore_index=True)

n_pos = labeled["label"].sum()
n_neg = len(labeled) - n_pos
print(f"Labeled pairs: {len(labeled):,}")
print(f"  Positives: {n_pos:,} ({100*n_pos/len(labeled):.1f}%)")
print(f"  Negatives: {n_neg:,} ({100*n_neg/len(labeled):.1f}%)")

# Entity-level split (no pair-level leakage)
train_ids, val_ids = group_split_entities(labeled["source1_entity_id"], VAL_FRAC, RANDOM_SEED)
train_df = labeled[labeled["source1_entity_id"].isin(train_ids)]
val_df   = labeled[labeled["source1_entity_id"].isin(val_ids)]
print(f"\\nEntity split: train={len(train_ids):,}  val={len(val_ids):,} entities")
print(f"  Train pairs: {len(train_df):,} (pos={train_df['label'].sum():,})")
print(f"  Val pairs:   {len(val_df):,} (pos={val_df['label'].sum():,})")

feat_cols = [c for c in FEATURE_COLUMNS if c in labeled.columns]
if "ann_score" in labeled.columns and "ann_score" not in feat_cols:
    feat_cols.append("ann_score")
print(f"\\nFeature columns for model: {feat_cols}")
"""))

# =========================================================================
# CELL 18: LightGBM training
# =========================================================================
cells.append(md("## 17. Train LightGBM"))
cells.append(code("""dtrain = lgb.Dataset(train_df[feat_cols], label=train_df["label"])
dval   = lgb.Dataset(val_df[feat_cols], label=val_df["label"], reference=dtrain)

params = {
    "objective": "binary",
    "metric": "auc",
    "learning_rate": 0.05,
    "num_leaves": 63,
    "min_data_in_leaf": 50,
    "feature_fraction": 0.85,
    "bagging_fraction": 0.85,
    "bagging_freq": 5,
    "is_unbalance": True,
    "seed": RANDOM_SEED,
    "verbosity": -1,
}

print("Training LightGBM...")
booster = lgb.train(
    params, dtrain, num_boost_round=2000, valid_sets=[dval],
    callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)],
)

# Feature importance
imp = pd.DataFrame({"feature": feat_cols, "importance": booster.feature_importance()})
imp = imp.sort_values("importance", ascending=False)
print("\\nFeature importance:")
for _, row in imp.head(15).iterrows():
    print(f"  {row['feature']:30s}  {int(row['importance']):>6}")
"""))

# =========================================================================
# CELL 19: Threshold tuning
# =========================================================================
cells.append(md("## 18. Threshold Tuning (Macro F0.5)"))
cells.append(code("""print("Scoring validation candidates...")
val_cand2 = feat2[feat2["source1_entity_id"].isin(val_ids)]
val_cand3 = feat3[feat3["source1_entity_id"].isin(val_ids)]
val_cand = pd.concat([val_cand2, val_cand3], ignore_index=True)
val_cand["score"] = booster.predict(val_cand[feat_cols])

print("Sweeping thresholds...")
best_t, best_f, curve = best_threshold_for_f05(val_cand, gt_map, val_ids)

print(f"\\nBest threshold: {best_t:.4f}")
print(f"Best val macro F0.5: {best_f:.4f}")
"""))

# =========================================================================
# CELL 20: Save artifacts
# =========================================================================
cells.append(md("## 19. Save Model Artifacts"))
cells.append(code("""booster.save_model(os.path.join(OUT_DIR, "model.txt"))
imp.to_csv(os.path.join(OUT_DIR, "feature_importance.csv"), index=False)

with open(os.path.join(OUT_DIR, "threshold.json"), "w") as f:
    json.dump({
        "threshold": best_t,
        "val_f05": best_f,
        "curve": curve,
        "feature_cols": feat_cols,
    }, f, indent=2)

elapsed = time.time() - t_start
print(f"Training complete in {elapsed:.0f}s ({elapsed/60:.1f}min)")
print(f"Artifacts saved to: {OUT_DIR}")
print(f"  model.txt")
print(f"  threshold.json")
print(f"  feature_importance.csv")
"""))

# =========================================================================
# CELL 21: Inference header
# =========================================================================
cells.append(md("""---
## 20. Inference on Test Data

Run this section to generate the submission files.
"""))

cells.append(code("""print("Loading test data...")
test_s1 = normalize_dataframe(pd.read_csv(os.path.join(TEST_DIR, "test_source1.tsv"), sep="\\t"))
test_s2 = normalize_dataframe(pd.read_csv(os.path.join(TEST_DIR, "test_source2.tsv"), sep="\\t"))
test_s3 = normalize_dataframe(pd.read_csv(os.path.join(TEST_DIR, "test_source3.tsv"), sep="\\t"))

valid_s2_ids = set(test_s2["entity_id"])
valid_s3_ids = set(test_s3["entity_id"])
all_test_s1_ids = test_s1["entity_id"].tolist()
print(f"  Test S1={len(test_s1):,}  S2={len(test_s2):,}  S3={len(test_s3):,}")
"""))

cells.append(code("""# Blocking
print("Test blocking S1 x S2...")
test_tok2 = generate_token_candidates(test_s1, test_s2) if USE_TOKEN_BLOCKING else pd.DataFrame(columns=["source1_entity_id", "other_entity_id"])
print("Test blocking S1 x S3...")
test_tok3 = generate_token_candidates(test_s1, test_s3) if USE_TOKEN_BLOCKING else pd.DataFrame(columns=["source1_entity_id", "other_entity_id"])

test_ann2 = pd.DataFrame(columns=["source1_entity_id", "other_entity_id", "ann_score"])
test_ann3 = pd.DataFrame(columns=["source1_entity_id", "other_entity_id", "ann_score"])
if USE_ANN:
    print("Test ANN S1 x S2...")
    test_ann2 = generate_embedding_candidates(test_s1, test_s2)
    print("Test ANN S1 x S3...")
    test_ann3 = generate_embedding_candidates(test_s1, test_s3)

# Fair capping
if not test_tok2.empty: test_tok2 = test_tok2.assign(_q=quick_score(test_tok2, test_s1, test_s2))
if not test_tok3.empty: test_tok3 = test_tok3.assign(_q=quick_score(test_tok3, test_s1, test_s3))
test_cand2 = cap_candidates_fair(test_tok2, test_ann2, score_col_lex="_q")
test_cand3 = cap_candidates_fair(test_tok3, test_ann3, score_col_lex="_q")
if "_q" in test_cand2.columns: test_cand2 = test_cand2.drop(columns="_q")
if "_q" in test_cand3.columns: test_cand3 = test_cand3.drop(columns="_q")

# Filter to valid IDs
test_cand2 = test_cand2[test_cand2["other_entity_id"].isin(valid_s2_ids)]
test_cand3 = test_cand3[test_cand3["other_entity_id"].isin(valid_s3_ids)]
print(f"Test candidates: S2={len(test_cand2):,}  S3={len(test_cand3):,}")
"""))

cells.append(code("""# Features + scoring
print("Test features S2...")
test_feat2 = build_pair_features(test_cand2, test_s1, test_s2)
print("Test features S3...")
test_feat3 = build_pair_features(test_cand3, test_s1, test_s3)

# Ensure all feature columns exist
for col in feat_cols:
    if col not in test_feat2.columns: test_feat2[col] = 0.0
    if col not in test_feat3.columns: test_feat3[col] = 0.0

test_feat2["score"] = booster.predict(test_feat2[feat_cols])
test_feat3["score"] = booster.predict(test_feat3[feat_cols])
test_all = pd.concat([test_feat2, test_feat3], ignore_index=True)

# Load threshold
with open(os.path.join(OUT_DIR, "threshold.json")) as f:
    meta = json.load(f)
threshold = meta["threshold"]
print(f"Using threshold: {threshold:.4f}")

# Build output maps
candidate_map = test_all.groupby("source1_entity_id")["other_entity_id"].apply(list).to_dict()
matched = test_all[test_all["score"] >= threshold]
matched_map = matched.groupby("source1_entity_id")["other_entity_id"].apply(list).to_dict()
print(f"Predicted matches: {sum(len(v) for v in matched_map.values()):,}")
"""))

cells.append(code("""# Write output TSVs
OUTPUT_DIR = os.path.join(OUT_DIR, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def write_tsv(path, s1_ids, id_map):
    col2 = "matched_entity_ids" if "matching" in path else "candidate_entity_ids"
    with open(path, "w") as f:
        f.write(f"source1_entity_id\\t{col2}\\n")
        for sid in s1_ids:
            ids = id_map.get(sid, [])
            seen = set()
            ordered = [i for i in ids if not (i in seen or seen.add(i))]
            f.write(f"{sid}\\t{','.join(ordered)}\\n")

write_tsv(os.path.join(OUTPUT_DIR, "candidate_pairs.tsv"), all_test_s1_ids, candidate_map)
write_tsv(os.path.join(OUTPUT_DIR, "matching_results.tsv"), all_test_s1_ids, matched_map)

print(f"Output written to {OUTPUT_DIR}/")
print(f"  candidate_pairs.tsv")
print(f"  matching_results.tsv")
print("\\nDownload these files for submission!")
"""))

# =========================================================================
# CELL 22: Download helper
# =========================================================================
cells.append(md("## 21. Download Results"))
cells.append(code("""# Create a zip of all outputs for easy download
import shutil
zip_path = os.path.join(OUT_DIR, "submission")
shutil.make_archive(zip_path, "zip", OUTPUT_DIR)
print(f"Submission zip: {zip_path}.zip")
print("Download this zip from the SageMaker file browser (left panel).")
"""))

# =========================================================================
# Build the notebook
# =========================================================================
notebook = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3 (ipykernel)",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.10.0",
            "codemirror_mode": {"name": "ipython", "version": 3},
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "pygments_lexer": "ipython3",
        }
    },
    "cells": cells,
}

out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sagemaker_pipeline.ipynb")
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=1, ensure_ascii=False)

print(f"Notebook written to: {out_path}")
print(f"  {len(cells)} cells ({sum(1 for c in cells if c['cell_type']=='code')} code, {sum(1 for c in cells if c['cell_type']=='markdown')} markdown)")
