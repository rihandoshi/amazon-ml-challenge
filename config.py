"""Shared config / constants for the entity resolution pipeline."""
import os

# ---------------------------------------------------------------------------
# Parallelism — auto-detect vCPUs, cap at 16 (ml.r5.4xlarge = 16 vCPUs)
# ---------------------------------------------------------------------------
NUM_WORKERS = min(os.cpu_count() or 4, 16)

# ---------------------------------------------------------------------------
# Checkpointing — enable stage-level resume on crash
# ---------------------------------------------------------------------------
CHECKPOINT_ENABLED = True

# ---------------------------------------------------------------------------
# Experiment / ablation flags — flip these to compare runs
# ---------------------------------------------------------------------------
USE_TOKEN_BLOCKING = True
USE_ANN = True                    # requires --use-embeddings CLI flag too
USE_SEPARATE_NAME_ANN = True      # ANN on name-only text (helps Devanagari→Latin)
USE_ADDRESS_ANN = False           # ANN on address-only text (experimental)
USE_CHAR_NGRAM_FEATURES = True    # char 3/4-gram cosine sim features
USE_ASYMMETRIC_E5_PREFIXES = True # "query:" for S1, "passage:" for S2/S3

# ---------------------------------------------------------------------------
# Token-blocking safety limits — prevent many-to-many memory explosion
# ---------------------------------------------------------------------------
# Skip any (country, token) block whose estimated pair count
# (s1_count × other_count) exceeds this.  Sensible range: 500–10_000.
MAX_TOKEN_BLOCK_PAIRS = 8_000

# Also skip any individual token that appears in > this many records on
# either side.  Acts as a backstop independent of the cross-product check.
MAX_TOKEN_FREQ = 2_000

# ---------------------------------------------------------------------------
# Candidate ranking / capping
# ---------------------------------------------------------------------------
# After blocking, keep the top-K candidates per S1 entity from each source.
# Lexical candidates (token/zip/city blocking) are ranked by name quick_score.
# ANN candidates are ranked by their ANN cosine similarity.
# The two pools are unioned, so an S1 entity can have up to LEX+ANN candidates.
CANDIDATE_CAP_LEXICAL = 40        # top-K from lexical/token blocking
CANDIDATE_CAP_ANN = 20            # top-K from ANN (guaranteed slots)

# ---------------------------------------------------------------------------
# ANN retrieval
# ---------------------------------------------------------------------------
ANN_TOP_K = 20                    # neighbours to retrieve per query (higher = better recall)

# ---------------------------------------------------------------------------
# Negative sampling
# ---------------------------------------------------------------------------
MAX_NEGATIVES_PER_ENTITY = 20     # hard negatives from blocking candidates

# ---------------------------------------------------------------------------
# Legal-suffix / common-word normalization for business names.
# Longer keys first is NOT required here because we match whole tokens after splitting.
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
    "the": "",  # drop leading/stray articles
}

# Tokens that carry almost no discriminative signal for blocking (stopword-ish
# for business names). NOT dropped from features, only from blocking keys.
NAME_STOPWORDS = {
    "the", "and", "of", "&", "a", "an", "for", "llc", "llp", "inc", "incorporated",
    "corp", "corporation", "ltd", "limited", "pvt", "private", "co", "company",
    "group", "services", "service", "international", "intl",
}

ADDRESS_TOKEN_MAP = {
    "rd": "road", "road": "road",
    "st": "street", "str": "street", "street": "street",
    "ave": "avenue", "av": "avenue", "avenue": "avenue",
    "blvd": "boulevard", "boulevard": "boulevard",
    "dr": "drive", "drive": "drive",
    "ln": "lane", "lane": "lane",
    "ct": "court", "court": "court",
    "pl": "place", "place": "place",
    "sq": "square", "square": "square",
    "apt": "apartment", "apartment": "apartment",
    "bldg": "building", "building": "building",
    "flr": "floor", "floor": "floor",
    "unit": "unit",
    "hwy": "highway", "highway": "highway",
    "pkwy": "parkway", "parkway": "parkway",
    "no": "number", "number": "number",
    "po": "postoffice",
    "near": "near",
    "opp": "opposite", "opposite": "opposite",
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

MODEL_ID_EMBEDDING = "intfloat/multilingual-e5-small"  # MIT license, ~118M params, handles Devanagari+Latin

RANDOM_SEED = 42
