"""Train the pairwise match/no-match classifier.

Usage:
    python train.py --data-dir /path/to/dataset/train --out-dir /path/to/artifacts \
        [--use-embeddings] [--max-negatives 20] [--val-frac 0.15]

Produces in --out-dir:
    model.txt            LightGBM booster
    threshold.json        tuned decision threshold + validation F0.5 + full curve
    feature_importance.csv
"""
import argparse
import json
import os
import time

import lightgbm as lgb
import pandas as pd

from normalize import normalize_dataframe
from blocking import (
    generate_token_candidates, generate_embedding_candidates,
    cap_candidates_per_entity, cap_candidates_fair,
)
from features import build_pair_features, FEATURE_COLUMNS, quick_score
from pairs_builder import build_labeled_pairs, group_split_entities
from metrics import (
    macro_f05, best_threshold_for_f05,
    candidate_recall, evaluate_candidate_recall_at_k,
)
from config import (
    RANDOM_SEED, USE_TOKEN_BLOCKING,
    CANDIDATE_CAP_LEXICAL, CANDIDATE_CAP_ANN,
    MAX_NEGATIVES_PER_ENTITY,
)


def load_source(data_dir, name):
    return pd.read_csv(os.path.join(data_dir, name), sep="\t")


def _build_gt_map(gt):
    """Build ground-truth lookup: S1 id -> set of matched entity ids."""
    gt_map = {}
    for sid, ids in zip(gt["source1_entity_id"], gt["matched_entity_ids"]):
        ids = str(ids).strip()
        gt_map[sid] = set(ids.split(",")) if ids else set()
    return gt_map


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True, help="dir with train_source1.tsv etc.")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--use-embeddings", action="store_true",
                     help="augment blocking with FAISS ANN (needs sentence-transformers+faiss)")
    ap.add_argument("--max-negatives", type=int, default=MAX_NEGATIVES_PER_ENTITY)
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--max-candidates-lexical", type=int, default=CANDIDATE_CAP_LEXICAL,
                     help="max lexical candidates per entity (guaranteed slots)")
    ap.add_argument("--max-candidates-ann", type=int, default=CANDIDATE_CAP_ANN,
                     help="max ANN candidates per entity (guaranteed slots)")
    # Legacy flag for backward compatibility
    ap.add_argument("--max-candidates-per-entity", type=int, default=None,
                     help="(legacy) if set, uses old-style single-pool capping")
    ap.add_argument("--skip-candidate-recall-eval", action="store_true",
                     help="skip the candidate recall evaluation (faster but less diagnostic)")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    t_start = time.time()

    # =========================================================================
    # LOAD + NORMALIZE
    # =========================================================================
    print("=" * 60)
    print("Loading + normalizing...")
    s1 = normalize_dataframe(load_source(args.data_dir, "train_source1.tsv"))
    s2 = normalize_dataframe(load_source(args.data_dir, "train_source2.tsv"))
    s3 = normalize_dataframe(load_source(args.data_dir, "train_source3.tsv"))
    gt = load_source(args.data_dir, "train_ground_truth.tsv")
    gt["matched_entity_ids"] = gt["matched_entity_ids"].fillna("")
    gt_map = _build_gt_map(gt)

    print(f"\n  === Dataset Statistics ===")
    print(f"  S1 count:  {len(s1):,}")
    print(f"  S2 count:  {len(s2):,}")
    print(f"  S3 count:  {len(s3):,}")
    print(f"  GT entries: {len(gt):,}")
    n_with_matches = sum(1 for v in gt_map.values() if v)
    n_singletons = sum(1 for v in gt_map.values() if not v)
    total_true_matches = sum(len(v) for v in gt_map.values())
    print(f"  Entities with matches: {n_with_matches:,}")
    print(f"  Singleton entities:    {n_singletons:,}")
    print(f"  Total true match pairs: {total_true_matches:,}")

    for src_name, src_df in [("S1", s1), ("S2", s2), ("S3", s3)]:
        countries = src_df["country"].value_counts()
        print(f"  {src_name} countries: {dict(countries)}")

    # =========================================================================
    # CANDIDATE GENERATION (BLOCKING)
    # =========================================================================
    print("\n" + "=" * 60)
    print("Blocking S1 x S2...")
    t0 = time.time()
    token_cand2 = generate_token_candidates(s1, s2) if USE_TOKEN_BLOCKING else \
        pd.DataFrame(columns=["source1_entity_id", "other_entity_id"])
    print(f"  Token blocking S1xS2: {len(token_cand2):,} candidates ({time.time()-t0:.1f}s)")

    print("Blocking S1 x S3...")
    t0 = time.time()
    token_cand3 = generate_token_candidates(s1, s3) if USE_TOKEN_BLOCKING else \
        pd.DataFrame(columns=["source1_entity_id", "other_entity_id"])
    print(f"  Token blocking S1xS3: {len(token_cand3):,} candidates ({time.time()-t0:.1f}s)")

    ann2 = pd.DataFrame(columns=["source1_entity_id", "other_entity_id", "ann_score"])
    ann3 = pd.DataFrame(columns=["source1_entity_id", "other_entity_id", "ann_score"])

    if args.use_embeddings:
        print("\nEmbedding ANN S1 x S2...")
        t0 = time.time()
        ann2 = generate_embedding_candidates(s1, s2)
        print(f"  ANN S1xS2: {len(ann2):,} candidates ({time.time()-t0:.1f}s)")

        print("Embedding ANN S1 x S3...")
        t0 = time.time()
        ann3 = generate_embedding_candidates(s1, s3)
        print(f"  ANN S1xS3: {len(ann3):,} candidates ({time.time()-t0:.1f}s)")

    # =========================================================================
    # CANDIDATE CAPPING (FAIR: separate lexical + ANN pools)
    # =========================================================================
    print("\n" + "=" * 60)
    print("Candidate ranking and capping...")

    if args.max_candidates_per_entity is not None:
        # Legacy mode: single-pool capping (backward compat)
        print("  Using legacy single-pool capping (--max-candidates-per-entity)")
        cand2 = pd.concat([token_cand2, ann2[["source1_entity_id", "other_entity_id"]]]).drop_duplicates()
        cand3 = pd.concat([token_cand3, ann3[["source1_entity_id", "other_entity_id"]]]).drop_duplicates()
        cand2 = cand2.assign(_q=quick_score(cand2, s1, s2))
        cand2 = cap_candidates_per_entity(cand2, "_q", args.max_candidates_per_entity).drop(columns="_q")
        cand3 = cand3.assign(_q=quick_score(cand3, s1, s3))
        cand3 = cap_candidates_per_entity(cand3, "_q", args.max_candidates_per_entity).drop(columns="_q")
    else:
        # Fair capping: separate pools for lexical and ANN
        # Score lexical candidates with cheap name similarity
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
        # Drop temporary score column
        if "_q" in cand2.columns:
            cand2 = cand2.drop(columns="_q")
        if "_q" in cand3.columns:
            cand3 = cand3.drop(columns="_q")

    print(f"\n  Candidates after capping: S2={len(cand2):,}  S3={len(cand3):,}")
    total_cand = len(cand2) + len(cand3)
    print(f"  Total candidates: {total_cand:,}")

    # =========================================================================
    # CANDIDATE RECALL EVALUATION
    # =========================================================================
    if not args.skip_candidate_recall_eval:
        print("\n" + "=" * 60)
        print("Candidate recall evaluation...")

        all_cand = pd.concat([cand2, cand3], ignore_index=True)
        cr = candidate_recall(all_cand, gt_map)
        print(f"\n  Overall candidate recall: {cr['candidate_recall']:.4f}")
        print(f"  Found {cr['found_matches']:,} / {cr['total_true_matches']:,} true matches")
        print(f"  Entities fully recalled: {cr['entities_fully_recalled']:,} / {cr['entities_with_matches']:,}")
        print(f"  Entities with missed matches: {cr['entities_with_missed']:,}")

        # Evaluate at different K values (on a subset if dataset is huge)
        if len(all_cand) > 0:
            # Score all candidates for K evaluation
            score_col = "ann_score" if "ann_score" in all_cand.columns else None
            if score_col is None:
                all_cand_scored = all_cand.assign(
                    _eval_score=quick_score(all_cand, s1,
                                             pd.concat([s2, s3], ignore_index=True))
                )
                score_col = "_eval_score"
            else:
                all_cand_scored = all_cand.copy()
                # Fill missing ann_score with quick_score for evaluation
                if all_cand_scored["ann_score"].isna().any():
                    s23 = pd.concat([s2, s3], ignore_index=True)
                    missing_mask = all_cand_scored["ann_score"].isna()
                    if missing_mask.any():
                        missing_pairs = all_cand_scored[missing_mask]
                        all_cand_scored.loc[missing_mask, "ann_score"] = \
                            quick_score(missing_pairs, s1, s23).values / 100.0

            evaluate_candidate_recall_at_k(
                all_cand_scored, score_col, gt_map,
                k_values=(20, 50, 100, 200, 500),
            )

    # =========================================================================
    # FEATURE ENGINEERING
    # =========================================================================
    print("\n" + "=" * 60)
    print("Building features...")
    t0 = time.time()
    feat2 = build_pair_features(cand2, s1, s2)
    feat3 = build_pair_features(cand3, s1, s3)
    print(f"  Feature engineering: {time.time()-t0:.1f}s")
    print(f"  Feature columns: {len([c for c in FEATURE_COLUMNS if c in feat2.columns])}")

    # =========================================================================
    # LABELING (hard negatives from blocking)
    # =========================================================================
    print("\n" + "=" * 60)
    print("Labeling...")
    labeled2 = build_labeled_pairs(feat2, gt, max_negatives_per_entity=args.max_negatives)
    labeled3 = build_labeled_pairs(feat3, gt, max_negatives_per_entity=args.max_negatives)
    labeled = pd.concat([labeled2, labeled3], ignore_index=True)
    n_pos = labeled["label"].sum()
    n_neg = len(labeled) - n_pos
    print(f"  Labeled pairs: {len(labeled):,}")
    print(f"  Positives: {n_pos:,} ({100*n_pos/len(labeled):.1f}%)")
    print(f"  Negatives: {n_neg:,} ({100*n_neg/len(labeled):.1f}%)")
    print(f"  Positive:Negative ratio: 1:{n_neg/max(n_pos,1):.1f}")

    # =========================================================================
    # ENTITY-LEVEL TRAIN/VAL SPLIT (no pair-level leakage)
    # =========================================================================
    print("\n" + "=" * 60)
    train_ids, val_ids = group_split_entities(labeled["source1_entity_id"], args.val_frac, RANDOM_SEED)
    train_df = labeled[labeled["source1_entity_id"].isin(train_ids)]
    val_df = labeled[labeled["source1_entity_id"].isin(val_ids)]
    print(f"  Entity-level split: train={len(train_ids):,} val={len(val_ids):,} entities")
    print(f"  Train pairs: {len(train_df):,} (pos={train_df['label'].sum():,})")
    print(f"  Val pairs:   {len(val_df):,} (pos={val_df['label'].sum():,})")

    feat_cols = [c for c in FEATURE_COLUMNS if c in labeled.columns]
    # Also include ann_score if present as a feature
    if "ann_score" in labeled.columns and "ann_score" not in feat_cols:
        feat_cols.append("ann_score")

    print(f"  Feature columns used: {feat_cols}")

    dtrain = lgb.Dataset(train_df[feat_cols], label=train_df["label"])
    dval = lgb.Dataset(val_df[feat_cols], label=val_df["label"], reference=dtrain)

    # =========================================================================
    # LIGHTGBM TRAINING
    # =========================================================================
    print("\n" + "=" * 60)
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
    booster.save_model(os.path.join(args.out_dir, "model.txt"))

    imp = pd.DataFrame({"feature": feat_cols, "importance": booster.feature_importance()})
    imp.sort_values("importance", ascending=False).to_csv(
        os.path.join(args.out_dir, "feature_importance.csv"), index=False)
    print("\n  Feature importance (top 10):")
    for _, row in imp.sort_values("importance", ascending=False).head(10).iterrows():
        print(f"    {row['feature']:30s}  {int(row['importance']):>6}")

    # =========================================================================
    # THRESHOLD TUNING on validation candidates
    # =========================================================================
    print("\n" + "=" * 60)
    print("Threshold tuning on validation split...")
    # For an honest F0.5, score ALL val-entity candidates (not just the
    # capped/labeled subsample used for training), including their true
    # singletons, so recall-from-blocking limitations show up correctly.
    val_s1_ids = val_ids
    val_cand2 = feat2[feat2["source1_entity_id"].isin(val_s1_ids)]
    val_cand3 = feat3[feat3["source1_entity_id"].isin(val_s1_ids)]
    val_cand = pd.concat([val_cand2, val_cand3], ignore_index=True)
    val_cand["score"] = booster.predict(val_cand[feat_cols])

    best_t, best_f, curve = best_threshold_for_f05(val_cand, gt_map, val_s1_ids)
    print(f"\n  Best threshold={best_t:.4f}  val macro F0.5={best_f:.4f}")

    with open(os.path.join(args.out_dir, "threshold.json"), "w") as f:
        json.dump({
            "threshold": best_t,
            "val_f05": best_f,
            "curve": curve,
            "feature_cols": feat_cols,
        }, f, indent=2)

    # =========================================================================
    # FINAL SUMMARY
    # =========================================================================
    elapsed = time.time() - t_start
    print("\n" + "=" * 60)
    print(f"  DONE in {elapsed:.0f}s ({elapsed/60:.1f}min)")
    print(f"  Artifacts written to: {args.out_dir}")
    print(f"    model.txt")
    print(f"    threshold.json")
    print(f"    feature_importance.csv")
    print("=" * 60)


if __name__ == "__main__":
    main()
