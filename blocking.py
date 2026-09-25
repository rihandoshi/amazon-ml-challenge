"""Candidate generation (blocking).

Two complementary strategies, unioned together:

1. Token blocking (exact, vectorized via pandas merge): explode normalized
   name tokens (and separately zip codes, and city+first-token) into long
   frames and inner-join S1 against S2/S3 on shared keys within the same
   country. This is O(N) to build and catches "same words, different order /
   different noise" matches. It is the main recall driver and is cheap
   enough to run over the full 2M x 5M x 5.3M dataset on a single big
   instance if you partition by country (see `generate_candidates`).

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

from config import NAME_STOPWORDS


def _token_frame(df: pd.DataFrame, min_len: int = 3) -> pd.DataFrame:
    ex = df[["entity_id", "country", "name_tokens"]].explode("name_tokens")
    ex = ex.rename(columns={"name_tokens": "token"})
    ex = ex[ex["token"].notna() & (ex["token"].str.len() >= min_len)]
    ex = ex[~ex["token"].isin(NAME_STOPWORDS)]
    return ex


def _pairs_from_token_join(s1_tok: pd.DataFrame, other_tok: pd.DataFrame) -> pd.DataFrame:
    merged = s1_tok.merge(
        other_tok, on=["country", "token"], suffixes=("_s1", "_other")
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

        s1_tok = _token_frame(s1_c)
        other_tok = _token_frame(other_c)
        all_pairs.append(_pairs_from_token_join(s1_tok, other_tok))
        all_pairs.append(_pairs_from_zip_join(s1_c, other_c))
        all_pairs.append(_pairs_from_city_firsttoken_join(s1_c, other_c))

    if not all_pairs:
        return pd.DataFrame(columns=["source1_entity_id", "other_entity_id"])
    out = pd.concat(all_pairs, ignore_index=True).drop_duplicates()
    return out


# ---------------------------------------------------------------------------
# Optional embedding-ANN augmentation stage
# ---------------------------------------------------------------------------

def generate_embedding_candidates(s1_df: pd.DataFrame, other_df: pd.DataFrame,
                                   top_k: int = 10, model_id: str = None,
                                   batch_size: int = 512) -> pd.DataFrame:
    """FAISS ANN search over sentence-embeddings of `norm_name + norm_addr`,
    partitioned by country. Requires `sentence-transformers` and `faiss-cpu`
    (or `faiss-gpu` on a SageMaker GPU instance) — install separately, it's
    intentionally not in the light requirements used for unit-testing the
    rest of this pipeline.

    This is what catches the ~5% of Indian S2 records (and ~3% of S3) whose
    business_name is in Devanagari while S1's is always Latin script, plus
    any other case with zero shared name token.
    """
    import numpy as np
    from sentence_transformers import SentenceTransformer
    import faiss

    from config import MODEL_ID_EMBEDDING
    model = SentenceTransformer(model_id or MODEL_ID_EMBEDDING)

    all_pairs = []
    for c in sorted(set(s1_df["country"].unique()) | set(other_df["country"].unique())):
        s1_c = s1_df[s1_df["country"] == c]
        other_c = other_df[other_df["country"] == c]
        if s1_c.empty or other_c.empty:
            continue

        def embed(texts):
            # e5 models expect a "query: " / "passage: " prefix for best results
            prefixed = [f"passage: {t}" for t in texts]
            return model.encode(prefixed, batch_size=batch_size, show_progress_bar=False,
                                 normalize_embeddings=True, convert_to_numpy=True)

        other_text = (other_c["norm_name"] + " " + other_c["norm_addr"]).tolist()
        other_emb = embed(other_text).astype("float32")

        index = faiss.IndexFlatIP(other_emb.shape[1])  # cosine sim via normalized inner product
        index.add(other_emb)

        s1_text = (s1_c["norm_name"] + " " + s1_c["norm_addr"]).tolist()
        s1_emb = embed(s1_text).astype("float32")

        sims, idxs = index.search(s1_emb, min(top_k, len(other_c)))

        s1_ids = s1_c["entity_id"].to_numpy()
        other_ids = other_c["entity_id"].to_numpy()
        for i in range(len(s1_ids)):
            for j, sim in zip(idxs[i], sims[i]):
                if j < 0:
                    continue
                all_pairs.append((s1_ids[i], other_ids[j], float(sim)))

    return pd.DataFrame(all_pairs, columns=["source1_entity_id", "other_entity_id", "ann_score"])


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
