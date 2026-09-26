"""Run the full pipeline on the test set and write the two required outputs.

Usage:
    python infer.py --data-dir /path/to/dataset/test --model-dir /path/to/artifacts \
        --out-dir /path/to/output [--use-embeddings] [--threshold 0.5] \
        [--resume] [--fresh]

Writes, tab-separated, exactly per the spec:
    <out-dir>/candidate_pairs.tsv      (last blocking-stage candidates, pre-threshold)
    <out-dir>/matching_results.tsv     (final matches, post-threshold)

Every Source-1 test entity gets exactly one row in both files (empty string
for no candidates / no matches), matching_entity_ids only ever reference
S2-/S3- ids present in the test files, and matches are a subset of candidates.

Checkpoints are written to <out-dir>/checkpoints/ after each major stage so
that a crash allows resuming from the last completed stage (pass --resume).
"""
import argparse
import json
import os
import pickle
import time

import lightgbm as lgb
import pandas as pd

from normalize import normalize_dataframe
from blocking import (
    generate_token_candidates, generate_embedding_candidates,
    cap_candidates_per_entity, cap_candidates_fair,
)
from features import build_pair_features, quick_score
from config import (
    USE_TOKEN_BLOCKING,
    CANDIDATE_CAP_LEXICAL, CANDIDATE_CAP_ANN,
    CHECKPOINT_ENABLED,
)


# ---------------------------------------------------------------------------
# Checkpoint helpers (same as train.py — lightweight, no external dependency)
# ---------------------------------------------------------------------------

def _ckpt_path(ckpt_dir, name):
    return os.path.join(ckpt_dir, name)


def _save_checkpoint(ckpt_dir, name, data):
    os.makedirs(ckpt_dir, exist_ok=True)
    path = _ckpt_path(ckpt_dir, name)
    with open(path, "wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    size_mb = os.path.getsize(path) / (1024 * 1024)
    print(f"  [CHECKPOINT] Saved '{name}' ({size_mb:.1f} MB)")


def _load_checkpoint(ckpt_dir, name):
    path = _ckpt_path(ckpt_dir, name)
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        data = pickle.load(f)
    size_mb = os.path.getsize(path) / (1024 * 1024)
    print(f"  [CHECKPOINT] Loaded '{name}' ({size_mb:.1f} MB)")
    return data


def _has_checkpoint(ckpt_dir, name):
    return os.path.exists(_ckpt_path(ckpt_dir, name))


def _clear_checkpoints(ckpt_dir):
    if os.path.exists(ckpt_dir):
        for f in os.listdir(ckpt_dir):
            if f.startswith("ckpt_"):
                os.remove(os.path.join(ckpt_dir, f))
        print("  [CHECKPOINT] Cleared all checkpoints")


def load_source(data_dir, name):
    return pd.read_csv(os.path.join(data_dir, name), sep="\t")


def write_id_list_tsv(path, s1_ids, id_map: dict):
    with open(path, "w") as f:
        f.write("source1_entity_id\tmatched_entity_ids\n" if "matching" in path
                 else "source1_entity_id\tcandidate_entity_ids\n")
        for sid in s1_ids:
            ids = id_map.get(sid, [])
            # de-dup while preserving order, just in case
            seen = set()
            ordered = [i for i in ids if not (i in seen or seen.add(i))]
            f.write(f"{sid}\t{','.join(ordered)}\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True, help="dir with test_source1.tsv etc.")
    ap.add_argument("--model-dir", required=True, help="dir with model.txt + threshold.json from train.py")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--use-embeddings", action="store_true", default=True,
                     help="augment blocking with FAISS ANN (default: enabled)")
    ap.add_argument("--no-embeddings", action="store_true",
                     help="disable ANN embeddings for a faster token-only run")
    ap.add_argument("--threshold", type=float, default=None, help="override the tuned threshold")
    ap.add_argument("--max-candidates-lexical", type=int, default=CANDIDATE_CAP_LEXICAL)
    ap.add_argument("--max-candidates-ann", type=int, default=CANDIDATE_CAP_ANN)
    # Legacy flag for backward compatibility
    ap.add_argument("--max-candidates-per-entity", type=int, default=None)
    # Checkpoint / resume flags
    ap.add_argument("--resume", action="store_true", default=CHECKPOINT_ENABLED,
                     help="resume from last checkpoint if available (default: enabled)")
    ap.add_argument("--fresh", action="store_true",
                     help="ignore existing checkpoints and start fresh")
    ap.add_argument("--checkpoint-dir", type=str, default=None,
                     help="directory for checkpoints (default: <out-dir>/checkpoints/)")
    args = ap.parse_args()
    if args.no_embeddings:
        args.use_embeddings = False
    os.makedirs(args.out_dir, exist_ok=True)

    ckpt_dir = args.checkpoint_dir or os.path.join(args.out_dir, "checkpoints")
    use_resume = args.resume and not args.fresh

    if args.fresh:
        _clear_checkpoints(ckpt_dir)

    if use_resume:
        print(f"  [CHECKPOINT] Resume enabled — checkpoints in: {ckpt_dir}")

    t_start = time.time()

    # =========================================================================
    # STAGE 1: LOAD + NORMALIZE
    # =========================================================================
    if use_resume and _has_checkpoint(ckpt_dir, "ckpt_infer_normalized.pkl"):
        ckpt = _load_checkpoint(ckpt_dir, "ckpt_infer_normalized.pkl")
        s1, s2, s3 = ckpt["s1"], ckpt["s2"], ckpt["s3"]
        print("  Loaded normalized test data from checkpoint")
    else:
        print("Loading + normalizing test data...")
        s1 = normalize_dataframe(load_source(args.data_dir, "test_source1.tsv"))
        s2 = normalize_dataframe(load_source(args.data_dir, "test_source2.tsv"))
        s3 = normalize_dataframe(load_source(args.data_dir, "test_source3.tsv"))

        if use_resume:
            _save_checkpoint(ckpt_dir, "ckpt_infer_normalized.pkl", {
                "s1": s1, "s2": s2, "s3": s3,
            })

    valid_s2_ids = set(s2["entity_id"])
    valid_s3_ids = set(s3["entity_id"])
    all_s1_ids = s1["entity_id"].tolist()

    print(f"  S1={len(s1):,}  S2={len(s2):,}  S3={len(s3):,}")

    # Log any new countries not seen in training (informational)
    known_countries = {"US", "India"}  # Countries we have state extraction for
    test_countries = set(s1["country"].unique()) | set(s2["country"].unique()) | set(s3["country"].unique())
    new_countries = test_countries - known_countries - {"UNK"}
    if new_countries:
        print(f"  NOTE: New countries in test data (no state extraction): {new_countries}")
        print(f"    → Blocking/features will still work via name tokens, zip, and embeddings")

    # =========================================================================
    # STAGE 2: BLOCKING + CAPPING
    # =========================================================================
    if use_resume and _has_checkpoint(ckpt_dir, "ckpt_infer_candidates.pkl"):
        ckpt = _load_checkpoint(ckpt_dir, "ckpt_infer_candidates.pkl")
        cand2, cand3 = ckpt["cand2"], ckpt["cand3"]
        print("  Loaded candidates from checkpoint")
    else:
        print("Blocking...")
        token_cand2 = generate_token_candidates(s1, s2) if USE_TOKEN_BLOCKING else \
            pd.DataFrame(columns=["source1_entity_id", "other_entity_id"])
        token_cand3 = generate_token_candidates(s1, s3) if USE_TOKEN_BLOCKING else \
            pd.DataFrame(columns=["source1_entity_id", "other_entity_id"])

        ann2 = pd.DataFrame(columns=["source1_entity_id", "other_entity_id", "ann_score"])
        ann3 = pd.DataFrame(columns=["source1_entity_id", "other_entity_id", "ann_score"])

        if args.use_embeddings:
            print("Embedding ANN S1 x S2...")
            ann2 = generate_embedding_candidates(s1, s2)
            print("Embedding ANN S1 x S3...")
            ann3 = generate_embedding_candidates(s1, s3)

        # --- Fair candidate capping ---
        if args.max_candidates_per_entity is not None:
            # Legacy mode
            cand2 = pd.concat([token_cand2, ann2[["source1_entity_id", "other_entity_id"]]]).drop_duplicates()
            cand3 = pd.concat([token_cand3, ann3[["source1_entity_id", "other_entity_id"]]]).drop_duplicates()
            cand2 = cand2.assign(_q=quick_score(cand2, s1, s2))
            cand2 = cap_candidates_per_entity(cand2, "_q", args.max_candidates_per_entity).drop(columns="_q")
            cand3 = cand3.assign(_q=quick_score(cand3, s1, s3))
            cand3 = cap_candidates_per_entity(cand3, "_q", args.max_candidates_per_entity).drop(columns="_q")
        else:
            if not token_cand2.empty:
                token_cand2 = token_cand2.assign(_q=quick_score(token_cand2, s1, s2))
            if not token_cand3.empty:
                token_cand3 = token_cand3.assign(_q=quick_score(token_cand3, s1, s3))

            cand2 = cap_candidates_fair(
                token_cand2, ann2,
                score_col_lex="_q", score_col_ann="ann_score",
                max_lex=args.max_candidates_lexical, max_ann=args.max_candidates_ann,
            )
            cand3 = cap_candidates_fair(
                token_cand3, ann3,
                score_col_lex="_q", score_col_ann="ann_score",
                max_lex=args.max_candidates_lexical, max_ann=args.max_candidates_ann,
            )
            if "_q" in cand2.columns:
                cand2 = cand2.drop(columns="_q")
            if "_q" in cand3.columns:
                cand3 = cand3.drop(columns="_q")

        # Sanity: candidates must only reference ids that actually exist in test files
        cand2 = cand2[cand2["other_entity_id"].isin(valid_s2_ids)]
        cand3 = cand3[cand3["other_entity_id"].isin(valid_s3_ids)]

        if use_resume:
            _save_checkpoint(ckpt_dir, "ckpt_infer_candidates.pkl", {
                "cand2": cand2, "cand3": cand3,
            })

    print(f"  Candidates after capping: S2={len(cand2):,}  S3={len(cand3):,}")

    # =========================================================================
    # STAGE 3: FEATURIZE + SCORE
    # =========================================================================
    if use_resume and _has_checkpoint(ckpt_dir, "ckpt_infer_features.pkl"):
        ckpt = _load_checkpoint(ckpt_dir, "ckpt_infer_features.pkl")
        feat2, feat3 = ckpt["feat2"], ckpt["feat3"]
        print("  Loaded features from checkpoint")
    else:
        print("Featurizing...")
        feat2 = build_pair_features(cand2, s1, s2)
        feat3 = build_pair_features(cand3, s1, s3)

        if use_resume:
            _save_checkpoint(ckpt_dir, "ckpt_infer_features.pkl", {
                "feat2": feat2, "feat3": feat3,
            })

    with open(os.path.join(args.model_dir, "threshold.json")) as f:
        meta = json.load(f)
    threshold = args.threshold if args.threshold is not None else meta["threshold"]
    feat_cols = meta["feature_cols"]
    print(f"  Using threshold={threshold:.4f}")

    booster = lgb.Booster(model_file=os.path.join(args.model_dir, "model.txt"))

    # Ensure feature columns exist (fill missing features with 0)
    # This handles the case where test data has different characteristics
    # than training data (e.g., new countries, missing features)
    for col in feat_cols:
        if col not in feat2.columns:
            feat2[col] = 0.0
            print(f"  WARNING: Feature '{col}' missing in S2 features, filled with 0")
        if col not in feat3.columns:
            feat3[col] = 0.0
            print(f"  WARNING: Feature '{col}' missing in S3 features, filled with 0")

    feat2["score"] = booster.predict(feat2[feat_cols])
    feat3["score"] = booster.predict(feat3[feat_cols])

    all_feat = pd.concat([feat2, feat3], ignore_index=True)

    candidate_map = (
        all_feat.groupby("source1_entity_id")["other_entity_id"].apply(list).to_dict()
    )
    matched = all_feat[all_feat["score"] >= threshold]
    matched_map = (
        matched.groupby("source1_entity_id")["other_entity_id"].apply(list).to_dict()
    )

    n_matched = sum(len(v) for v in matched_map.values())
    print(f"  Total predicted matches: {n_matched:,}")

    write_id_list_tsv(os.path.join(args.out_dir, "candidate_pairs.tsv"), all_s1_ids, candidate_map)
    write_id_list_tsv(os.path.join(args.out_dir, "matching_results.tsv"), all_s1_ids, matched_map)

    elapsed = time.time() - t_start
    print(f"\n  Done in {elapsed:.0f}s")
    print("Wrote", os.path.join(args.out_dir, "candidate_pairs.tsv"))
    print("Wrote", os.path.join(args.out_dir, "matching_results.tsv"))


if __name__ == "__main__":
    main()
