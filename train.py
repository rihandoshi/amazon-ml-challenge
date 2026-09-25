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

import lightgbm as lgb
import pandas as pd

from normalize import normalize_dataframe
from blocking import generate_token_candidates, generate_embedding_candidates, cap_candidates_per_entity
from features import build_pair_features, FEATURE_COLUMNS, quick_score
from pairs_builder import build_labeled_pairs, group_split_entities
from metrics import macro_f05, best_threshold_for_f05
from config import RANDOM_SEED


def load_source(data_dir, name):
    return pd.read_csv(os.path.join(data_dir, name), sep="\t")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True, help="dir with train_source1.tsv etc.")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--use-embeddings", action="store_true",
                     help="augment blocking with FAISS ANN (needs sentence-transformers+faiss)")
    ap.add_argument("--max-negatives", type=int, default=20)
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--max-candidates-per-entity", type=int, default=50)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    print("Loading + normalizing...")
    s1 = normalize_dataframe(load_source(args.data_dir, "train_source1.tsv"))
    s2 = normalize_dataframe(load_source(args.data_dir, "train_source2.tsv"))
    s3 = normalize_dataframe(load_source(args.data_dir, "train_source3.tsv"))
    gt = load_source(args.data_dir, "train_ground_truth.tsv")
    gt["matched_entity_ids"] = gt["matched_entity_ids"].fillna("")

    print("Blocking S1 x S2...")
    cand2 = generate_token_candidates(s1, s2)
    print("Blocking S1 x S3...")
    cand3 = generate_token_candidates(s1, s3)

    if args.use_embeddings:
        print("Embedding ANN S1 x S2...")
        ann2 = generate_embedding_candidates(s1, s2)
        cand2 = pd.concat([cand2, ann2[["source1_entity_id", "other_entity_id"]]]).drop_duplicates()
        print("Embedding ANN S1 x S3...")
        ann3 = generate_embedding_candidates(s1, s3)
        cand3 = pd.concat([cand3, ann3[["source1_entity_id", "other_entity_id"]]]).drop_duplicates()

    if args.max_candidates_per_entity:
        cand2 = cand2.assign(_q=quick_score(cand2, s1, s2))
        cand2 = cap_candidates_per_entity(cand2, "_q", args.max_candidates_per_entity).drop(columns="_q")
        cand3 = cand3.assign(_q=quick_score(cand3, s1, s3))
        cand3 = cap_candidates_per_entity(cand3, "_q", args.max_candidates_per_entity).drop(columns="_q")

    print(f"Candidates: S2={len(cand2)} S3={len(cand3)}")

    print("Building features...")
    feat2 = build_pair_features(cand2, s1, s2)
    feat3 = build_pair_features(cand3, s1, s3)

    print("Labeling...")
    labeled2 = build_labeled_pairs(feat2, gt, max_negatives_per_entity=args.max_negatives)
    labeled3 = build_labeled_pairs(feat3, gt, max_negatives_per_entity=args.max_negatives)
    labeled = pd.concat([labeled2, labeled3], ignore_index=True)
    print(f"Labeled pairs: {len(labeled)} (positives={labeled['label'].sum()})")

    train_ids, val_ids = group_split_entities(labeled["source1_entity_id"], args.val_frac, RANDOM_SEED)
    train_df = labeled[labeled["source1_entity_id"].isin(train_ids)]
    val_df = labeled[labeled["source1_entity_id"].isin(val_ids)]

    feat_cols = [c for c in FEATURE_COLUMNS if c in labeled.columns]
    dtrain = lgb.Dataset(train_df[feat_cols], label=train_df["label"])
    dval = lgb.Dataset(val_df[feat_cols], label=val_df["label"], reference=dtrain)

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

    # ---- threshold tuning on a *candidate-set* view of the val split ----
    # For an honest F0.5, score ALL val-entity candidates (not just the
    # capped/labeled subsample used for training), including their true
    # singletons, so recall-from-blocking limitations show up correctly.
    val_s1_ids = val_ids
    val_cand2 = feat2[feat2["source1_entity_id"].isin(val_s1_ids)]
    val_cand3 = feat3[feat3["source1_entity_id"].isin(val_s1_ids)]
    val_cand = pd.concat([val_cand2, val_cand3], ignore_index=True)
    val_cand["score"] = booster.predict(val_cand[feat_cols])

    gt_map = {sid: set(ids.split(",")) if ids else set()
              for sid, ids in zip(gt["source1_entity_id"], gt["matched_entity_ids"])}

    best_t, best_f, curve = best_threshold_for_f05(val_cand, gt_map, val_s1_ids)
    print(f"Best threshold={best_t:.2f}  val macro F0.5={best_f:.4f}")

    with open(os.path.join(args.out_dir, "threshold.json"), "w") as f:
        json.dump({"threshold": best_t, "val_f05": best_f, "curve": curve, "feature_cols": feat_cols}, f, indent=2)

    print("Done. Artifacts written to", args.out_dir)


if __name__ == "__main__":
    main()
