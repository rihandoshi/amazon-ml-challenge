"""Pairwise feature engineering.

Takes a `candidate_pairs` frame (source1_entity_id, other_entity_id) plus the
normalized S1 / other-source dataframes, and returns one feature row per pair.
Vectorized with rapidfuzz.process / cdist where possible; falls back to
row-wise apply only where necessary (address component comparisons).

Parallelized: feature computation is split across CPU cores via joblib for
large pair sets (>10k pairs).
"""
import numpy as np
import pandas as pd
from rapidfuzz import fuzz, distance
from joblib import Parallel, delayed

from config import USE_CHAR_NGRAM_FEATURES, NUM_WORKERS


FEATURE_COLUMNS = [
    "name_jaro_winkler", "name_levenshtein_ratio", "name_token_sort_ratio",
    "name_token_set_ratio", "name_partial_ratio", "name_len_diff",
    "name_common_token_frac", "name_first_token_match",
    "addr_token_sort_ratio", "addr_partial_ratio",
    "city_exact_match", "city_fuzzy_score", "state_exact_match",
    "zip_exact_match", "zip_present_both", "house_no_exact_match",
    "country_match",
    # Char n-gram cosine similarity features (added when USE_CHAR_NGRAM_FEATURES=True)
    "name_char_3gram_cosine", "name_char_4gram_cosine",
    "addr_char_3gram_cosine",
]


def _safe(s):
    return s if isinstance(s, str) else ""


def _char_ngrams(text: str, n: int) -> dict:
    """Extract character n-grams from text and return as a frequency dict.

    Efficient implementation that avoids building large TF-IDF matrices:
    computes n-grams only for the specific pair being compared.
    """
    if not text or len(text) < n:
        return {}
    grams = {}
    for i in range(len(text) - n + 1):
        gram = text[i:i + n]
        grams[gram] = grams.get(gram, 0) + 1
    return grams


def _ngram_cosine_sim(text_a: str, text_b: str, n: int) -> float:
    """Cosine similarity between char n-gram frequency vectors of two strings.

    Efficient pairwise computation: no TF-IDF matrix needed, just counts
    for the two strings being compared.

    Captures: typos, punctuation diffs, spacing, abbreviations,
    transliteration differences, small spelling variations.
    """
    if not text_a or not text_b:
        return 0.0

    grams_a = _char_ngrams(text_a, n)
    grams_b = _char_ngrams(text_b, n)

    if not grams_a or not grams_b:
        return 0.0

    # Compute cosine similarity using only shared keys (sparse dot product)
    common_keys = set(grams_a.keys()) & set(grams_b.keys())
    if not common_keys:
        return 0.0

    dot = sum(grams_a[k] * grams_b[k] for k in common_keys)
    norm_a = sum(v * v for v in grams_a.values()) ** 0.5
    norm_b = sum(v * v for v in grams_b.values()) ** 0.5

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return dot / (norm_a * norm_b)


def _compute_features_chunk(
    left_chunk, right_chunk, left_addr_chunk, right_addr_chunk,
    left_tokens_chunk, right_tokens_chunk,
    left_city_chunk, right_city_chunk,
    use_ngram_features,
):
    """Compute features for a chunk of pairs (called in parallel)."""
    n = len(left_chunk)
    jaro = np.empty(n); lev = np.empty(n); tsort = np.empty(n)
    tset = np.empty(n); partial = np.empty(n); addr_tsort = np.empty(n)
    addr_partial = np.empty(n); common_frac = np.empty(n); first_tok = np.empty(n)
    city_fuzzy = np.empty(n)

    name_3gram = np.zeros(n)
    name_4gram = np.zeros(n)
    addr_3gram = np.zeros(n)

    for i in range(n):
        a, b = _safe(left_chunk[i]), _safe(right_chunk[i])
        jaro[i] = distance.JaroWinkler.similarity(a, b) if a and b else 0.0
        lev[i] = fuzz.ratio(a, b) / 100.0
        tsort[i] = fuzz.token_sort_ratio(a, b) / 100.0
        tset[i] = fuzz.token_set_ratio(a, b) / 100.0
        partial[i] = fuzz.partial_ratio(a, b) / 100.0

        aa, ab = _safe(left_addr_chunk[i]), _safe(right_addr_chunk[i])
        addr_tsort[i] = fuzz.token_sort_ratio(aa, ab) / 100.0 if aa and ab else 0.0
        addr_partial[i] = fuzz.partial_ratio(aa, ab) / 100.0 if aa and ab else 0.0

        lt = left_tokens_chunk[i] if isinstance(left_tokens_chunk[i], list) else []
        rt = right_tokens_chunk[i] if isinstance(right_tokens_chunk[i], list) else []
        if lt and rt:
            common = len(set(lt) & set(rt))
            common_frac[i] = common / max(len(set(lt) | set(rt)), 1)
            first_tok[i] = 1.0 if lt[0] == rt[0] else 0.0
        else:
            common_frac[i] = 0.0
            first_tok[i] = 0.0

        lc, rc = left_city_chunk[i], right_city_chunk[i]
        city_fuzzy[i] = fuzz.ratio(_safe(lc), _safe(rc)) / 100.0 if lc and rc else 0.0

        if use_ngram_features:
            name_3gram[i] = _ngram_cosine_sim(a, b, 3)
            name_4gram[i] = _ngram_cosine_sim(a, b, 4)
            addr_3gram[i] = _ngram_cosine_sim(aa, ab, 3)

    return {
        "jaro": jaro, "lev": lev, "tsort": tsort, "tset": tset, "partial": partial,
        "addr_tsort": addr_tsort, "addr_partial": addr_partial,
        "common_frac": common_frac, "first_tok": first_tok, "city_fuzzy": city_fuzzy,
        "name_3gram": name_3gram, "name_4gram": name_4gram, "addr_3gram": addr_3gram,
    }


def build_pair_features(pairs: pd.DataFrame, s1_df: pd.DataFrame, other_df: pd.DataFrame,
                         n_workers: int = None) -> pd.DataFrame:
    """`pairs` needs columns source1_entity_id, other_entity_id.
    `s1_df` / `other_df` need the columns produced by normalize.normalize_dataframe.

    Parallelized across CPU cores for large pair sets.
    Robust to unseen entity IDs (fills NaN with safe defaults).
    """
    if n_workers is None:
        n_workers = NUM_WORKERS

    s1_idx = s1_df.set_index("entity_id")
    other_idx = other_df.set_index("entity_id")

    # Safe .map() with .fillna() to handle unseen entity IDs gracefully
    left = pairs["source1_entity_id"].map(s1_idx["norm_name"]).fillna("")
    right = pairs["other_entity_id"].map(other_idx["norm_name"]).fillna("")
    left_addr = pairs["source1_entity_id"].map(s1_idx["norm_addr"]).fillna("")
    right_addr = pairs["other_entity_id"].map(other_idx["norm_addr"]).fillna("")

    left_tokens = pairs["source1_entity_id"].map(s1_idx["name_tokens"])
    right_tokens = pairs["other_entity_id"].map(other_idx["name_tokens"])
    # Replace NaN tokens with empty lists
    left_tokens = left_tokens.apply(lambda x: x if isinstance(x, list) else [])
    right_tokens = right_tokens.apply(lambda x: x if isinstance(x, list) else [])

    left_city = pairs["source1_entity_id"].map(s1_idx["city"])
    right_city = pairs["other_entity_id"].map(other_idx["city"])
    left_state = pairs["source1_entity_id"].map(s1_idx["state"])
    right_state = pairs["other_entity_id"].map(other_idx["state"])
    left_zip = pairs["source1_entity_id"].map(s1_idx["zip"])
    right_zip = pairs["other_entity_id"].map(other_idx["zip"])
    left_house = pairs["source1_entity_id"].map(s1_idx["house_no"])
    right_house = pairs["other_entity_id"].map(other_idx["house_no"])
    left_country = pairs["source1_entity_id"].map(s1_idx["country"]).fillna("UNK")
    right_country = pairs["other_entity_id"].map(other_idx["country"]).fillna("UNK")

    n = len(pairs)

    # Convert to lists for chunking
    left_l = left.tolist(); right_l = right.tolist()
    left_addr_l = left_addr.tolist(); right_addr_l = right_addr.tolist()
    left_tokens_l = left_tokens.tolist(); right_tokens_l = right_tokens.tolist()
    left_city_l = left_city.tolist(); right_city_l = right_city.tolist()

    # ---- Parallel feature computation ----
    if n > 10_000 and n_workers > 1:
        chunk_size = (n + n_workers - 1) // n_workers
        results = Parallel(n_jobs=n_workers, backend="loky")(
            delayed(_compute_features_chunk)(
                left_l[start:start + chunk_size],
                right_l[start:start + chunk_size],
                left_addr_l[start:start + chunk_size],
                right_addr_l[start:start + chunk_size],
                left_tokens_l[start:start + chunk_size],
                right_tokens_l[start:start + chunk_size],
                left_city_l[start:start + chunk_size],
                right_city_l[start:start + chunk_size],
                USE_CHAR_NGRAM_FEATURES,
            )
            for start in range(0, n, chunk_size)
        )
        # Concatenate chunk results
        jaro = np.concatenate([r["jaro"] for r in results])
        lev = np.concatenate([r["lev"] for r in results])
        tsort = np.concatenate([r["tsort"] for r in results])
        tset = np.concatenate([r["tset"] for r in results])
        partial = np.concatenate([r["partial"] for r in results])
        addr_tsort = np.concatenate([r["addr_tsort"] for r in results])
        addr_partial = np.concatenate([r["addr_partial"] for r in results])
        common_frac = np.concatenate([r["common_frac"] for r in results])
        first_tok = np.concatenate([r["first_tok"] for r in results])
        city_fuzzy = np.concatenate([r["city_fuzzy"] for r in results])
        name_3gram = np.concatenate([r["name_3gram"] for r in results])
        name_4gram = np.concatenate([r["name_4gram"] for r in results])
        addr_3gram = np.concatenate([r["addr_3gram"] for r in results])
    else:
        # Single-process fallback for small datasets
        result = _compute_features_chunk(
            left_l, right_l, left_addr_l, right_addr_l,
            left_tokens_l, right_tokens_l,
            left_city_l, right_city_l,
            USE_CHAR_NGRAM_FEATURES,
        )
        jaro = result["jaro"]; lev = result["lev"]; tsort = result["tsort"]
        tset = result["tset"]; partial = result["partial"]
        addr_tsort = result["addr_tsort"]; addr_partial = result["addr_partial"]
        common_frac = result["common_frac"]; first_tok = result["first_tok"]
        city_fuzzy = result["city_fuzzy"]
        name_3gram = result["name_3gram"]; name_4gram = result["name_4gram"]
        addr_3gram = result["addr_3gram"]

    out = pd.DataFrame({
        "source1_entity_id": pairs["source1_entity_id"].values,
        "other_entity_id": pairs["other_entity_id"].values,
        "name_jaro_winkler": jaro,
        "name_levenshtein_ratio": lev,
        "name_token_sort_ratio": tsort,
        "name_token_set_ratio": tset,
        "name_partial_ratio": partial,
        "name_len_diff": np.abs(pd.Series(left_l).str.len().fillna(0).values
                                 - pd.Series(right_l).str.len().fillna(0).values),
        "name_common_token_frac": common_frac,
        "name_first_token_match": first_tok,
        "addr_token_sort_ratio": addr_tsort,
        "addr_partial_ratio": addr_partial,
        "city_exact_match": (left_city.fillna("__L") == right_city.fillna("__R")).astype(int).values,
        "city_fuzzy_score": city_fuzzy,
        "state_exact_match": (left_state.fillna("__L") == right_state.fillna("__R")).astype(int).values,
        "zip_exact_match": ((left_zip.notna()) & (right_zip.notna()) & (left_zip == right_zip)).astype(int).values,
        "zip_present_both": ((left_zip.notna()) & (right_zip.notna())).astype(int).values,
        "house_no_exact_match": ((left_house.notna()) & (right_house.notna())
                                  & (left_house == right_house)).astype(int).values,
        "country_match": (left_country.values == right_country.values).astype(int),
        "name_char_3gram_cosine": name_3gram,
        "name_char_4gram_cosine": name_4gram,
        "addr_char_3gram_cosine": addr_3gram,
    })

    if "ann_score" in pairs.columns:
        out["ann_score"] = pairs["ann_score"].values

    # Carry over provenance columns if present
    for col in ["from_token_block", "from_ann"]:
        if col in pairs.columns:
            out[col] = pairs[col].values

    return out


def _quick_score_chunk(left_chunk, right_chunk):
    """Score a chunk of pairs (used in parallel quick_score)."""
    return [fuzz.token_set_ratio(a, b) for a, b in zip(left_chunk, right_chunk)]


def quick_score(pairs: pd.DataFrame, s1_df: pd.DataFrame, other_df: pd.DataFrame,
                n_workers: int = None) -> pd.Series:
    """Cheap single-number similarity used only to rank/cap candidates before
    full feature engineering (see blocking.cap_candidates_per_entity) —
    NOT the final model score.

    Parallelized for large pair sets (>10k).
    Robust to unseen entity IDs (fills NaN with empty string).
    """
    if n_workers is None:
        n_workers = NUM_WORKERS

    s1_idx = s1_df.set_index("entity_id")["norm_name"]
    other_idx = other_df.set_index("entity_id")["norm_name"]
    left = pairs["source1_entity_id"].map(s1_idx).fillna("").tolist()
    right = pairs["other_entity_id"].map(other_idx).fillna("").tolist()

    n = len(left)
    if n > 10_000 and n_workers > 1:
        chunk_size = (n + n_workers - 1) // n_workers
        results = Parallel(n_jobs=n_workers, backend="loky")(
            delayed(_quick_score_chunk)(
                left[start:start + chunk_size],
                right[start:start + chunk_size],
            )
            for start in range(0, n, chunk_size)
        )
        scores = []
        for r in results:
            scores.extend(r)
    else:
        scores = [fuzz.token_set_ratio(a, b) for a, b in zip(left, right)]

    return pd.Series(scores, index=pairs.index)
