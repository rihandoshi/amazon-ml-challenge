"""Pairwise feature engineering.

Takes a `candidate_pairs` frame (source1_entity_id, other_entity_id) plus the
normalized S1 / other-source dataframes, and returns one feature row per pair.
Vectorized with rapidfuzz.process / cdist where possible; falls back to
row-wise apply only where necessary (address component comparisons).
"""
import numpy as np
import pandas as pd
from rapidfuzz import fuzz, distance


FEATURE_COLUMNS = [
    "name_jaro_winkler", "name_levenshtein_ratio", "name_token_sort_ratio",
    "name_token_set_ratio", "name_partial_ratio", "name_len_diff",
    "name_common_token_frac", "name_first_token_match",
    "addr_token_sort_ratio", "addr_partial_ratio",
    "city_exact_match", "city_fuzzy_score", "state_exact_match",
    "zip_exact_match", "zip_present_both", "house_no_exact_match",
    "country_match",
]


def _safe(s):
    return s if isinstance(s, str) else ""


def build_pair_features(pairs: pd.DataFrame, s1_df: pd.DataFrame, other_df: pd.DataFrame) -> pd.DataFrame:
    """`pairs` needs columns source1_entity_id, other_entity_id.
    `s1_df` / `other_df` need the columns produced by normalize.normalize_dataframe.
    """
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

    left = left.tolist(); right = right.tolist()
    left_addr = left_addr.tolist(); right_addr = right_addr.tolist()
    left_tokens = left_tokens.tolist(); right_tokens = right_tokens.tolist()
    left_city_l = left_city.tolist(); right_city_l = right_city.tolist()

    for i in range(n):
        a, b = _safe(left[i]), _safe(right[i])
        jaro[i] = distance.JaroWinkler.similarity(a, b) if a and b else 0.0
        lev[i] = fuzz.ratio(a, b) / 100.0
        tsort[i] = fuzz.token_sort_ratio(a, b) / 100.0
        tset[i] = fuzz.token_set_ratio(a, b) / 100.0
        partial[i] = fuzz.partial_ratio(a, b) / 100.0

        aa, ab = _safe(left_addr[i]), _safe(right_addr[i])
        addr_tsort[i] = fuzz.token_sort_ratio(aa, ab) / 100.0 if aa and ab else 0.0
        addr_partial[i] = fuzz.partial_ratio(aa, ab) / 100.0 if aa and ab else 0.0

        lt = left_tokens[i] if isinstance(left_tokens[i], list) else []
        rt = right_tokens[i] if isinstance(right_tokens[i], list) else []
        if lt and rt:
            common = len(set(lt) & set(rt))
            common_frac[i] = common / max(len(set(lt) | set(rt)), 1)
            first_tok[i] = 1.0 if lt[0] == rt[0] else 0.0
        else:
            common_frac[i] = 0.0
            first_tok[i] = 0.0

        lc, rc = left_city_l[i], right_city_l[i]
        city_fuzzy[i] = fuzz.ratio(_safe(lc), _safe(rc)) / 100.0 if lc and rc else 0.0

    out = pd.DataFrame({
        "source1_entity_id": pairs["source1_entity_id"].values,
        "other_entity_id": pairs["other_entity_id"].values,
        "name_jaro_winkler": jaro,
        "name_levenshtein_ratio": lev,
        "name_token_sort_ratio": tsort,
        "name_token_set_ratio": tset,
        "name_partial_ratio": partial,
        "name_len_diff": np.abs(pd.Series(left).str.len().fillna(0).values
                                 - pd.Series(right).str.len().fillna(0).values),
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
    })

    if "ann_score" in pairs.columns:
        out["ann_score"] = pairs["ann_score"].values

    return out


def quick_score(pairs: pd.DataFrame, s1_df: pd.DataFrame, other_df: pd.DataFrame) -> pd.Series:
    """Cheap single-number similarity used only to rank/cap candidates before
    full feature engineering (see blocking.cap_candidates_per_entity) —
    NOT the final model score.
    """
    s1_idx = s1_df.set_index("entity_id")["norm_name"]
    other_idx = other_df.set_index("entity_id")["norm_name"]
    left = pairs["source1_entity_id"].map(s1_idx).fillna("")
    right = pairs["other_entity_id"].map(other_idx).fillna("")
    return pd.Series(
        [fuzz.token_set_ratio(a, b) for a, b in zip(left, right)],
        index=pairs.index,
    )
