"""Candidate generation (blocking).

Two complementary strategies, unioned together:

1. Token blocking (exact, vectorized via pandas merge): explode normalized
   name tokens (and separately zip codes, and city+first-token) into long
   frames and inner-join S1 against S2/S3 on shared keys within the same
   country. This is O(N) to build and catches "same words, different order /
   different noise" matches. It is the main recall driver and is cheap
   enough to run over the full 2M x 5M x 5.3M dataset on a single big
   instance if you partition by country (see `generate_candidates`).

   **Safety**: frequency-aware blocking skips (country, token) blocks whose
   estimated pair count (s1_count × other_count) exceeds MAX_TOKEN_BLOCK_PAIRS,
   preventing the catastrophic 146 GiB memory explosion.

2. Embedding ANN (ANN = approximate nearest neighbour via FAISS), catching
   cases with *no* shared token at all: typos that change every token,
   transliteration (Devanagari S2/S3 name vs Latin S1 name), heavy
   abbreviation, or reordering beyond what token blocking's stopword
   filtering handles. This needs `sentence-transformers` + `faiss`, both
   MIT/Apache-licensed and open weight (<=1B params), so it does not
   conflict with the "MIT/Apache, <=8B params" model constraint.

Both stages emit (source1_entity_id, other_entity_id) pairs; the union,
deduplicated, is your `candidate_pairs.tsv` (after the feature+model stage
narrows candidates -> matches, per the problem statement: candidate_pairs.tsv
must be the *last* candidate set actually scored).
"""
import pandas as pd

from config import (
    NAME_STOPWORDS, MAX_TOKEN_BLOCK_PAIRS, MAX_TOKEN_FREQ,
    CANDIDATE_CAP_LEXICAL, CANDIDATE_CAP_ANN,
    USE_ASYMMETRIC_E5_PREFIXES, USE_SEPARATE_NAME_ANN, USE_ADDRESS_ANN,
    ANN_TOP_K,
)


def _token_frame(df: pd.DataFrame, min_len: int = 3) -> pd.DataFrame:
    ex = df[["entity_id", "country", "name_tokens"]].explode("name_tokens")
    ex = ex.rename(columns={"name_tokens": "token"})
    ex = ex[ex["token"].notna() & (ex["token"].str.len() >= min_len)]
    ex = ex[~ex["token"].isin(NAME_STOPWORDS)]
    return ex


def _pairs_from_token_join(s1_tok: pd.DataFrame, other_tok: pd.DataFrame,
                            max_block_pairs: int = MAX_TOKEN_BLOCK_PAIRS,
                            max_token_freq: int = MAX_TOKEN_FREQ) -> pd.DataFrame:
    """Frequency-aware token blocking: filters out (country, token) blocks
    whose estimated pair count exceeds `max_block_pairs` BEFORE performing
    the many-to-many merge, preventing catastrophic memory explosions.

    Also skips tokens that exceed `max_token_freq` on either side.

    Returns (source1_entity_id, other_entity_id) plus a from_token_block flag.
    """
    # --- Step 1: compute per-(country, token) frequency on each side ---
    s1_freq = (
        s1_tok.groupby(["country", "token"])["entity_id"]
        .nunique()
        .reset_index(name="s1_count")
    )
    other_freq = (
        other_tok.groupby(["country", "token"])["entity_id"]
        .nunique()
        .reset_index(name="other_count")
    )

    # --- Step 2: join frequencies to estimate block sizes ---
    block_stats = s1_freq.merge(other_freq, on=["country", "token"], how="inner")
    block_stats["est_pairs"] = block_stats["s1_count"] * block_stats["other_count"]

    # --- Step 3: filter safe blocks ---
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
    safe_pairs_est = int(safe["est_pairs"].sum()) if len(safe) > 0 else 0

    print(f"  Token blocking: {len(block_stats)} unique (country,token) blocks")
    print(f"  Skipped {n_skipped} oversized blocks (est. {pairs_avoided:,} pairs avoided)")
    print(f"  Proceeding with {len(safe)} safe blocks (est. {safe_pairs_est:,} pairs)")

    if safe.empty:
        return pd.DataFrame(columns=["source1_entity_id", "other_entity_id"])

    # --- Step 4: filter token frames to safe tokens, THEN merge ---
    safe_keys = safe[["country", "token"]]
    s1_safe = s1_tok.merge(safe_keys, on=["country", "token"], how="inner")
    other_safe = other_tok.merge(safe_keys, on=["country", "token"], how="inner")

    merged = s1_safe.merge(
        other_safe, on=["country", "token"], suffixes=("_s1", "_other")
    )
    pairs = merged[["entity_id_s1", "entity_id_other"]].drop_duplicates()
    pairs.columns = ["source1_entity_id", "other_entity_id"]
    return pairs


def _pairs_from_zip_join(s1: pd.DataFrame, other: pd.DataFrame) -> pd.DataFrame:
    a = s1[s1["zip"].notna()][["entity_id", "country", "zip"]]
    b = other[other["zip"].notna()][["entity_id", "country", "zip"]]
    if a.empty or b.empty:
        return pd.DataFrame(columns=["source1_entity_id", "other_entity_id"])
    merged = a.merge(b, on=["country", "zip"], suffixes=("_s1", "_other"))
    pairs = merged[["entity_id_s1", "entity_id_other"]].drop_duplicates()
    pairs.columns = ["source1_entity_id", "other_entity_id"]
    return pairs


def _pairs_from_city_firsttoken_join(s1: pd.DataFrame, other: pd.DataFrame) -> pd.DataFrame:
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


def generate_token_candidates(s1_df: pd.DataFrame, other_df: pd.DataFrame,
                               per_country: bool = True) -> pd.DataFrame:
    """Union of token / zip / city+first-token blocking, S1 vs one other source.

    Set per_country=True (default) to loop over countries and concat — keeps
    peak memory bounded, which matters once you scale to the real 2M x 5M+
    row files (do this on SageMaker with a large-memory instance, e.g.
    r6i.8xlarge, or push the loop body out to a Ray / multiprocessing pool).
    """
    all_pairs = []
    countries = sorted(set(s1_df["country"].unique()) | set(other_df["country"].unique())) \
        if per_country else [None]

    for c in countries:
        s1_c = s1_df[s1_df["country"] == c] if c is not None else s1_df
        other_c = other_df[other_df["country"] == c] if c is not None else other_df
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


# ---------------------------------------------------------------------------
# Optional embedding-ANN augmentation stage
# ---------------------------------------------------------------------------

def _embed_texts(model, texts, prefix, batch_size):
    """Embed a list of texts with a given prefix."""
    prefixed = [f"{prefix}{t}" for t in texts]
    return model.encode(prefixed, batch_size=batch_size, show_progress_bar=False,
                         normalize_embeddings=True, convert_to_numpy=True).astype("float32")


def _ann_search_single(s1_c, other_c, model, text_fn, top_k, batch_size,
                        use_asymmetric=USE_ASYMMETRIC_E5_PREFIXES):
    """Run ANN search for one country partition using a text extraction function.

    text_fn(df) -> list of strings to embed.
    Returns list of (s1_id, other_id, similarity) tuples.
    """
    import numpy as np
    import faiss

    query_prefix = "query: " if use_asymmetric else "passage: "
    passage_prefix = "passage: "

    other_text = text_fn(other_c)
    other_emb = _embed_texts(model, other_text, passage_prefix, batch_size)

    index = faiss.IndexFlatIP(other_emb.shape[1])  # cosine sim via normalized inner product
    index.add(other_emb)

    s1_text = text_fn(s1_c)
    s1_emb = _embed_texts(model, s1_text, query_prefix, batch_size)

    sims, idxs = index.search(s1_emb, min(top_k, len(other_c)))

    s1_ids = s1_c["entity_id"].to_numpy()
    other_ids = other_c["entity_id"].to_numpy()
    results = []
    for i in range(len(s1_ids)):
        for j, sim in zip(idxs[i], sims[i]):
            if j < 0:
                continue
            results.append((s1_ids[i], other_ids[j], float(sim)))
    return results


def generate_embedding_candidates(s1_df: pd.DataFrame, other_df: pd.DataFrame,
                                   top_k: int = None, model_id: str = None,
                                   batch_size: int = 512,
                                   use_name_ann: bool = None,
                                   use_addr_ann: bool = None) -> pd.DataFrame:
    """FAISS ANN search over sentence-embeddings, partitioned by country.

    Supports multiple retrieval modes (unioned):
    - Combined: norm_name + norm_addr (always on when this function is called)
    - Name-only: norm_name (if use_name_ann=True)
    - Address-only: norm_addr (if use_addr_ann=True)

    Uses asymmetric query/passage prefixes for E5 models when configured.
    """
    import numpy as np
    from sentence_transformers import SentenceTransformer

    from config import MODEL_ID_EMBEDDING
    model = SentenceTransformer(model_id or MODEL_ID_EMBEDDING)

    if top_k is None:
        top_k = ANN_TOP_K
    if use_name_ann is None:
        use_name_ann = USE_SEPARATE_NAME_ANN
    if use_addr_ann is None:
        use_addr_ann = USE_ADDRESS_ANN

    # Define text extraction functions
    def combined_text(df):
        return (df["norm_name"] + " " + df["norm_addr"]).tolist()

    def name_text(df):
        return df["norm_name"].tolist()

    def addr_text(df):
        return df["norm_addr"].tolist()

    retrieval_modes = [("combined", combined_text)]
    if use_name_ann:
        retrieval_modes.append(("name", name_text))
    if use_addr_ann:
        retrieval_modes.append(("address", addr_text))

    print(f"  ANN modes: {[m[0] for m in retrieval_modes]}, top_k={top_k}")

    all_pairs = []
    for c in sorted(set(s1_df["country"].unique()) | set(other_df["country"].unique())):
        s1_c = s1_df[s1_df["country"] == c]
        other_c = other_df[other_df["country"] == c]
        if s1_c.empty or other_c.empty:
            continue

        print(f"  ANN country={c}: S1={len(s1_c):,} Other={len(other_c):,}")

        for mode_name, text_fn in retrieval_modes:
            results = _ann_search_single(s1_c, other_c, model, text_fn, top_k, batch_size)
            all_pairs.extend(results)

    df = pd.DataFrame(all_pairs, columns=["source1_entity_id", "other_entity_id", "ann_score"])
    # Deduplicate, keeping the best ann_score for each pair
    if not df.empty:
        df = df.sort_values("ann_score", ascending=False).drop_duplicates(
            subset=["source1_entity_id", "other_entity_id"], keep="first"
        )
    print(f"  Total ANN candidates: {len(df):,}")
    return df


def cap_candidates_fair(token_pairs: pd.DataFrame, ann_pairs: pd.DataFrame,
                         score_col_lex: str, score_col_ann: str = "ann_score",
                         max_lex: int = None, max_ann: int = None) -> pd.DataFrame:
    """Fair candidate capping that gives both lexical and ANN candidates
    guaranteed slots in the retained set.

    Strategy:
        top max_lex lexical candidates (by cheap name score)
      + top max_ann ANN candidates (by ANN cosine similarity)
      → union (deduplicated)

    This prevents ANN candidates from being unfairly discarded just because
    their lexical name score is low (e.g. Devanagari names, heavy typos).

    Also preserves provenance columns: from_token_block, from_ann, ann_score.
    """
    if max_lex is None:
        max_lex = CANDIDATE_CAP_LEXICAL
    if max_ann is None:
        max_ann = CANDIDATE_CAP_ANN

    retained_parts = []

    # --- Lexical candidates: top-K by cheap name score ---
    if not token_pairs.empty and score_col_lex in token_pairs.columns:
        top_lex = (
            token_pairs.sort_values(score_col_lex, ascending=False)
            .groupby("source1_entity_id", group_keys=False)
            .head(max_lex)
        )
        top_lex = top_lex.assign(from_token_block=1)
        retained_parts.append(top_lex[["source1_entity_id", "other_entity_id",
                                        score_col_lex, "from_token_block"]])
    elif not token_pairs.empty:
        # no score column yet — keep all (shouldn't normally happen)
        token_pairs = token_pairs.assign(from_token_block=1)
        retained_parts.append(token_pairs[["source1_entity_id", "other_entity_id",
                                            "from_token_block"]])

    # --- ANN candidates: top-K by ANN score (guaranteed slots) ---
    if not ann_pairs.empty and score_col_ann in ann_pairs.columns:
        top_ann = (
            ann_pairs.sort_values(score_col_ann, ascending=False)
            .groupby("source1_entity_id", group_keys=False)
            .head(max_ann)
        )
        top_ann = top_ann.assign(from_ann=1)
        retained_parts.append(top_ann[["source1_entity_id", "other_entity_id",
                                        score_col_ann, "from_ann"]])

    if not retained_parts:
        return pd.DataFrame(columns=["source1_entity_id", "other_entity_id"])

    # --- Union and deduplicate ---
    combined = pd.concat(retained_parts, ignore_index=True)
    # For duplicate pairs, merge provenance info
    agg_cols = {"from_token_block": "max", "from_ann": "max"}
    if score_col_lex in combined.columns:
        agg_cols[score_col_lex] = "max"
    if score_col_ann in combined.columns:
        agg_cols[score_col_ann] = "max"
    # Only aggregate columns that actually exist
    agg_cols = {k: v for k, v in agg_cols.items() if k in combined.columns}

    if agg_cols:
        combined = combined.groupby(
            ["source1_entity_id", "other_entity_id"], as_index=False
        ).agg(agg_cols)
    else:
        combined = combined.drop_duplicates(subset=["source1_entity_id", "other_entity_id"])

    # Fill NaN provenance flags
    for col in ["from_token_block", "from_ann"]:
        if col in combined.columns:
            combined[col] = combined[col].fillna(0).astype(int)

    print(f"  Fair cap: {len(combined):,} candidates retained "
          f"(lex<={max_lex}/entity, ann<={max_ann}/entity)")

    return combined


# Legacy function kept for backward compatibility
def cap_candidates_per_entity(pairs: pd.DataFrame, score_col: str, max_per_entity: int = 50) -> pd.DataFrame:
    """Keep only the top-N candidates per source1_entity_id by `score_col`.

    Use this after scoring the union of token+ANN candidates with a cheap
    similarity (see features.quick_score) to bound how many pairs go into
    the (more expensive) full feature engineering + model scoring stage.
    """
    return (
        pairs.sort_values(score_col, ascending=False)
        .groupby("source1_entity_id", group_keys=False)
        .head(max_per_entity)
    )
