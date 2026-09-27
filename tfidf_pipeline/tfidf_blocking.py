import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import time


def get_sparse_top_k(A, B, top_k=20, batch_size=2000, min_score=0.1):
    """
    Finds top_k highest dot products between rows of A and rows of B.
    A and B are scipy.sparse.csr_matrix.
    Returns positional (row-in-A, row-in-B) indices, NOT entity ids.
    """
    B_T = B.T.tocsc()

    s1_indices = []
    other_indices = []
    scores = []

    n_rows = A.shape[0]
    for start in range(0, n_rows, batch_size):
        end = min(start + batch_size, n_rows)
        A_batch = A[start:end]

        sim_batch = A_batch.dot(B_T).tocsr()

        for i in range(sim_batch.shape[0]):
            r_start = sim_batch.indptr[i]
            r_end = sim_batch.indptr[i + 1]

            if r_end == r_start:
                continue

            row_data = sim_batch.data[r_start:r_end]
            row_cols = sim_batch.indices[r_start:r_end]

            mask = row_data >= min_score
            row_data = row_data[mask]
            row_cols = row_cols[mask]

            if len(row_data) > top_k:
                idx = np.argpartition(row_data, -top_k)[-top_k:]
                best_scores = row_data[idx]
                best_cols = row_cols[idx]
            else:
                best_scores = row_data
                best_cols = row_cols

            s1_idx = start + i
            s1_indices.extend([s1_idx] * len(best_scores))
            other_indices.extend(best_cols)
            scores.extend(best_scores)

    return s1_indices, other_indices, scores


# ---------------------------------------------------------------------------
# NEW: shared-vectorizer fit/transform, so you fit ONCE across S1+S2+S3
# instead of once per (S1, S2) and once per (S1, S3) call.
# ---------------------------------------------------------------------------

def fit_vectorizer(text_series_list, ngram_range=(3, 4), min_df=2, max_df=0.5):
    """Fit one char n-gram TF-IDF vectorizer across several text columns
    (typically [s1[text_col], s2[text_col], s3[text_col]]).
    """
    vectorizer = TfidfVectorizer(
        analyzer="char_wb", ngram_range=ngram_range, min_df=min_df, max_df=max_df,
        dtype=np.float32,
    )
    combined = pd.concat([s.fillna("") for s in text_series_list]).drop_duplicates()
    vectorizer.fit(combined)
    return vectorizer


def transform_texts(vectorizer, text_series):
    return vectorizer.transform(text_series.fillna(""))


def tfidf_blocking_by_country(s1_df, s1_mat, other_df, other_mat,
                               top_k=20, batch_size=2000, min_score=0.2):
    """Country-partitioned TF-IDF top-k, given ALREADY-TRANSFORMED sparse
    matrices (row-aligned with s1_df / other_df respectively).

    This is the actual fix over the old approach: instead of computing the
    full S1 x other dot product and filtering by country afterwards, we
    slice the precomputed sparse matrices by country mask (cheap — no
    re-tokenizing, no re-fitting) and only ever multiply same-country
    blocks. On a ~60/40 US/India split this roughly halves wasted dot
    product work.
    """
    t0 = time.time()
    parts = []
    countries = sorted(set(s1_df["country"].unique()) | set(other_df["country"].unique()))

    for c in countries:
        s1_mask = (s1_df["country"] == c).to_numpy()
        other_mask = (other_df["country"] == c).to_numpy()
        if not s1_mask.any() or not other_mask.any():
            continue

        s1_sub = s1_mat[s1_mask]
        other_sub = other_mat[other_mask]
        s1_ids_sub = s1_df.loc[s1_mask, "entity_id"].to_numpy()
        other_ids_sub = other_df.loc[other_mask, "entity_id"].to_numpy()

        s1_idx, other_idx, scores = get_sparse_top_k(
            s1_sub, other_sub, top_k=top_k, batch_size=batch_size, min_score=min_score
        )
        if not s1_idx:
            continue

        parts.append(pd.DataFrame({
            "source1_entity_id": s1_ids_sub[s1_idx],
            "other_entity_id": other_ids_sub[other_idx],
            "ann_score": scores,
        }))

    if not parts:
        return pd.DataFrame(columns=["source1_entity_id", "other_entity_id", "ann_score"])

    out = pd.concat(parts, ignore_index=True)
    print(f"  [TF-IDF] country-partitioned blocking done in {time.time()-t0:.1f}s. "
          f"Found {len(out):,} candidates across {len(countries)} countries.")
    return out


def tfidf_blocking(s1_df, other_df, text_col="norm_name", top_k=20, batch_size=2000,
                    min_score=0.2, vectorizer=None):
    """Backward-compatible single-pair entrypoint (same signature your
    test_tfidf.py already calls). Internally now does country-partitioned
    blocking. Pass a pre-fit `vectorizer` (from fit_vectorizer) to avoid
    re-fitting when you're calling this more than once against the same
    source (e.g. once for S2, once for S3) — see tfidf_blocking_multi below
    for the fully-optimized path used by train_fast.py.
    """
    t0 = time.time()
    s1_text = s1_df[text_col]
    other_text = other_df[text_col]

    if vectorizer is None:
        print(f"  [TF-IDF] Fitting on {text_col}...")
        vectorizer = fit_vectorizer([s1_text, other_text])

    print("  [TF-IDF] Transforming...")
    s1_mat = transform_texts(vectorizer, s1_text)
    other_mat = transform_texts(vectorizer, other_text)

    print("  [TF-IDF] Computing country-partitioned sparse top-K dot product...")
    pairs = tfidf_blocking_by_country(
        s1_df, s1_mat, other_df, other_mat, top_k=top_k, batch_size=batch_size, min_score=min_score
    )
    print(f"  [TF-IDF] Blocking {text_col} done in {time.time()-t0:.1f}s. Found {len(pairs):,} candidates.")
    return pairs


def tfidf_blocking_multi(s1_df, other_dfs: dict, text_col="norm_name", top_k=20,
                          batch_size=2000, min_score=0.2):
    """Fit ONE vectorizer across s1 + all of other_dfs.values(), transform
    each source once, then run country-partitioned blocking against each
    other source. This is what train_fast.py should use for S1 vs {S2, S3}
    instead of two independent tfidf_blocking() calls — avoids fitting the
    vectorizer twice and avoids re-transforming S1's text twice.

    other_dfs: dict like {"s2": s2_df, "s3": s3_df}
    Returns: dict like {"s2": pairs_df, "s3": pairs_df}
    """
    t0 = time.time()
    print(f"  [TF-IDF] Fitting shared vectorizer on {text_col} across "
          f"s1 + {list(other_dfs.keys())}...")
    all_texts = [s1_df[text_col]] + [df[text_col] for df in other_dfs.values()]
    vectorizer = fit_vectorizer(all_texts)

    s1_mat = transform_texts(vectorizer, s1_df[text_col])

    results = {}
    for name, df in other_dfs.items():
        print(f"  [TF-IDF] {name}: transforming + blocking...")
        other_mat = transform_texts(vectorizer, df[text_col])
        results[name] = tfidf_blocking_by_country(
            s1_df, s1_mat, df, other_mat, top_k=top_k, batch_size=batch_size, min_score=min_score
        )

    print(f"  [TF-IDF] tfidf_blocking_multi({text_col}) total: {time.time()-t0:.1f}s")
    return results
