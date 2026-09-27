"""
Fast pipeline using TF-IDF sparse dot product for blocking instead of embeddings.
This achieves high recall while running much faster and using less memory.

CHANGES from the original train_fast.py:
  - tfidf_blocking is now country-partitioned and fits its vectorizer ONCE
    across s1+s2+s3 instead of once per (s1,s2) and once per (s1,s3) call
    (see tfidf_blocking_multi in tfidf_blocking.py). This is a pure speedup,
    no behavior change to what gets found.
  - Optional second TF-IDF pass over `norm_addr` (ADDR_BLOCKING flag below),
    unioned into the same "ann" candidate pool before the fair cap. Catches
    matches where the name is mangled but the address is clean. Off is a
    one-line change if you're tight on time -- everything else is unaffected.
  - Inference (STAGE 5) refits its own vectorizer on the TEST text rather
    than reusing the training vectorizer -- TF-IDF blocking isn't a model
    feature, so there's no train/test consistency requirement, and refitting
    avoids losing recall on vocabulary specific to the test-only France rows.
"""
import argparse
import json
import os
import sys
import time
import pickle

import lightgbm as lgb
import numpy as np
import pandas as pd

# Add parent dir to path to reuse normalize, features, etc.
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)

from normalize import normalize_dataframe
from blocking import generate_token_candidates, cap_candidates_fair
from features import build_pair_features, FEATURE_COLUMNS, quick_score
from pairs_builder import build_labeled_pairs, group_split_entities
from metrics import best_threshold_for_f05, candidate_recall
from config import RANDOM_SEED, NUM_WORKERS

from tfidf_blocking import tfidf_blocking_multi

# --- Settings ---
# We cap S1 train set to 400k as requested to make it super fast.
S1_TRAIN_CAP = 400_000
# S2 and S3 caps are larger to ensure we find the matches for our S1 sample
S2_TRAIN_CAP = 1_000_000
S3_TRAIN_CAP = 1_000_000

CAP_LEXICAL = 40
CAP_TFIDF = 30
TFIDF_TOP_K = 30
TFIDF_MIN_SCORE = 0.2

# Second TF-IDF pass over normalized address text, unioned into the same
# "ann" candidate pool as the name-based TF-IDF before capping. Set to False
# if STAGE 2 / STAGE 5 timing is already tight -- it's a pure addition, no
# other code path depends on it.
ADDR_BLOCKING = True
ADDR_TOP_K = 15
ADDR_MIN_SCORE = 0.3


def _write_tsv(path: str, s1_ids: list, id_map: dict, col: str) -> None:
    with open(path, "w") as f:
        f.write(f"source1_entity_id\t{col}\n")
        for sid in s1_ids:
            ids = id_map.get(sid, [])
            seen, ordered = set(), []
            for i in ids:
                if i not in seen:
                    seen.add(i)
                    ordered.append(i)
            f.write(f"{sid}\t{','.join(ordered)}\n")


def _build_gt_map(gt: pd.DataFrame) -> dict:
    gt_map = {}
    for sid, ids in zip(gt["source1_entity_id"], gt["matched_entity_ids"]):
        ids = str(ids).strip()
        gt_map[sid] = set(ids.split(",")) if ids else set()
    return gt_map


def _union_ann_pools(name_pairs: pd.DataFrame, addr_pairs: pd.DataFrame) -> pd.DataFrame:
    """Union name-TFIDF and address-TFIDF candidate pools into one 'ann' pool,
    keeping the higher ann_score when a pair shows up in both.
    """
    if addr_pairs is None or addr_pairs.empty:
        return name_pairs
    if name_pairs is None or name_pairs.empty:
        return addr_pairs
    combined = pd.concat([name_pairs, addr_pairs], ignore_index=True)
    combined = combined.sort_values("ann_score", ascending=False).drop_duplicates(
        subset=["source1_entity_id", "other_entity_id"], keep="first"
    )
    return combined.reset_index(drop=True)


def _blocking_stage(s1, s2, s3, stage_name="Train"):
    """Runs token blocking + (name + optional address) TF-IDF blocking for
    S1 vs {S2, S3}, fitting each TF-IDF vectorizer ONCE across all three
    sources rather than once per pair. Returns cand2, cand3 (capped).
    """
    print(f"\n  --- {stage_name}: token blocking ---")
    tok2 = generate_token_candidates(s1, s2)
    tok3 = generate_token_candidates(s1, s3)

    print(f"  --- {stage_name}: name TF-IDF blocking (shared vectorizer) ---")
    name_results = tfidf_blocking_multi(
        s1, {"s2": s2, "s3": s3}, text_col="norm_name",
        top_k=TFIDF_TOP_K, min_score=TFIDF_MIN_SCORE,
    )
    tfidf2, tfidf3 = name_results["s2"], name_results["s3"]

    if ADDR_BLOCKING:
        print(f"  --- {stage_name}: address TF-IDF blocking (shared vectorizer) ---")
        addr_results = tfidf_blocking_multi(
            s1, {"s2": s2, "s3": s3}, text_col="norm_addr",
            top_k=ADDR_TOP_K, min_score=ADDR_MIN_SCORE,
        )
        tfidf2 = _union_ann_pools(tfidf2, addr_results["s2"])
        tfidf3 = _union_ann_pools(tfidf3, addr_results["s3"])

    if not tok2.empty:
        tok2 = tok2.assign(_q=quick_score(tok2, s1, s2))
    if not tok3.empty:
        tok3 = tok3.assign(_q=quick_score(tok3, s1, s3))

    cand2 = cap_candidates_fair(tok2, tfidf2, "_q", score_col_ann="ann_score",
                                 max_lex=CAP_LEXICAL, max_ann=CAP_TFIDF)
    cand3 = cap_candidates_fair(tok3, tfidf3, "_q", score_col_ann="ann_score",
                                 max_lex=CAP_LEXICAL, max_ann=CAP_TFIDF)

    for c in ["_q"]:
        if c in cand2.columns:
            cand2 = cand2.drop(columns=c)
        if c in cand3.columns:
            cand3 = cand3.drop(columns=c)

    return cand2, cand3


def run_pipeline(data_dir, out_dir):
    t0_total = time.time()
    os.makedirs(out_dir, exist_ok=True)
    train_dir = os.path.join(data_dir, "train")
    test_dir = os.path.join(data_dir, "test")

    print("\n============================================================")
    print("STAGE 1: Load, subsample & normalize training data")
    print("============================================================")

    t0 = time.time()
    s1_raw = pd.read_csv(os.path.join(train_dir, "train_source1.tsv"), sep="\t")
    s2_raw = pd.read_csv(os.path.join(train_dir, "train_source2.tsv"), sep="\t")
    s3_raw = pd.read_csv(os.path.join(train_dir, "train_source3.tsv"), sep="\t")
    gt = pd.read_csv(os.path.join(train_dir, "train_ground_truth.tsv"), sep="\t")
    gt["matched_entity_ids"] = gt["matched_entity_ids"].fillna("")

    # Subsample to speed up training while keeping GT rows
    gt_s1_ids = set(gt["source1_entity_id"])
    gt_s2_ids = set()
    gt_s3_ids = set()
    for ids in gt["matched_entity_ids"]:
        for eid in str(ids).split(","):
            eid = eid.strip()
            if eid.startswith("S2-"):
                gt_s2_ids.add(eid)
            elif eid.startswith("S3-"):
                gt_s3_ids.add(eid)

    def smart_sample(df, id_col, must_keep_ids, cap):
        if cap is None or len(df) <= cap:
            return df
        must = df[df[id_col].isin(must_keep_ids)]
        rest = df[~df[id_col].isin(must_keep_ids)]
        remaining = max(0, cap - len(must))
        filler = rest.sample(n=min(remaining, len(rest)), random_state=RANDOM_SEED)
        return pd.concat([must, filler], ignore_index=True)

    s1_raw = smart_sample(s1_raw, "entity_id", gt_s1_ids, S1_TRAIN_CAP)
    s2_raw = smart_sample(s2_raw, "entity_id", gt_s2_ids, S2_TRAIN_CAP)
    s3_raw = smart_sample(s3_raw, "entity_id", gt_s3_ids, S3_TRAIN_CAP)

    sampled_s1_ids = set(s1_raw["entity_id"])
    gt = gt[gt["source1_entity_id"].isin(sampled_s1_ids)].reset_index(drop=True)
    gt_map = _build_gt_map(gt)

    print(f"  Training sets: S1={len(s1_raw):,} S2={len(s2_raw):,} S3={len(s3_raw):,}")

    print("  Normalizing...")
    s1 = normalize_dataframe(s1_raw)
    s2 = normalize_dataframe(s2_raw)
    s3 = normalize_dataframe(s3_raw)
    print(f"  Normalize done ({time.time()-t0:.1f}s)")

    print("\n============================================================")
    print("STAGE 2: Blocking (Token + TF-IDF name" +
          (" + TF-IDF address" if ADDR_BLOCKING else "") + ") on Train")
    print("============================================================")
    t0 = time.time()

    cand2, cand3 = _blocking_stage(s1, s2, s3, stage_name="Train")
    print(f"  Train candidates generated: S2={len(cand2):,} S3={len(cand3):,} "
          f"({time.time()-t0:.1f}s)")

    # Blocking recall diagnostic
    cr = candidate_recall(pd.concat([cand2, cand3], ignore_index=True), gt_map)
    print(f"  Train Blocking recall: {cr['candidate_recall']:.4f}")

    print("\n============================================================")
    print("STAGE 3: Features & Labeling")
    print("============================================================")
    t0 = time.time()
    feat2 = build_pair_features(cand2, s1, s2)
    feat3 = build_pair_features(cand3, s1, s3)

    lab2 = build_labeled_pairs(feat2, gt)
    lab3 = build_labeled_pairs(feat3, gt)
    labeled = pd.concat([lab2, lab3], ignore_index=True)

    train_ids, val_ids = group_split_entities(labeled["source1_entity_id"], 0.15, RANDOM_SEED)
    train_df = labeled[labeled["source1_entity_id"].isin(train_ids)]
    val_df = labeled[labeled["source1_entity_id"].isin(val_ids)]

    feat_cols = [c for c in FEATURE_COLUMNS if c in labeled.columns]
    if "ann_score" in labeled.columns and "ann_score" not in feat_cols:
        feat_cols.append("ann_score")

    print(f"  Features built ({time.time()-t0:.1f}s). Train={len(train_df):,} Val={len(val_df):,}")

    print("\n============================================================")
    print("STAGE 4: LightGBM Training & Tuning")
    print("============================================================")
    dtrain = lgb.Dataset(train_df[feat_cols], label=train_df["label"])
    dval = lgb.Dataset(val_df[feat_cols], label=val_df["label"], reference=dtrain)
    params = {
        "objective": "binary", "metric": "auc", "learning_rate": 0.05,
        "num_leaves": 127, "min_data_in_leaf": 20, "feature_fraction": 0.85,
        "bagging_fraction": 0.85, "bagging_freq": 5, "is_unbalance": True,
        "seed": RANDOM_SEED, "verbosity": -1, "num_threads": NUM_WORKERS,
    }
    booster = lgb.train(
        params, dtrain, num_boost_round=1500, valid_sets=[dval],
        callbacks=[lgb.early_stopping(75), lgb.log_evaluation(100)],
    )
    booster.save_model(os.path.join(out_dir, "model_fast.txt"))

    val_cand = pd.concat([
        feat2[feat2["source1_entity_id"].isin(val_ids)],
        feat3[feat3["source1_entity_id"].isin(val_ids)],
    ], ignore_index=True)
    val_cand["score"] = booster.predict(val_cand[feat_cols])
    best_t, best_f, _ = best_threshold_for_f05(val_cand, gt_map, val_ids)
    print(f"  Best threshold={best_t:.4f}  val macro F0.5={best_f:.4f}")

    print("\n============================================================")
    print("STAGE 5: Inference on Test Set")
    print("============================================================")
    t0 = time.time()
    t1_raw = pd.read_csv(os.path.join(test_dir, "test_source1.tsv"), sep="\t")
    t2_raw = pd.read_csv(os.path.join(test_dir, "test_source2.tsv"), sep="\t")
    t3_raw = pd.read_csv(os.path.join(test_dir, "test_source3.tsv"), sep="\t")

    t1 = normalize_dataframe(t1_raw)
    t2 = normalize_dataframe(t2_raw)
    t3 = normalize_dataframe(t3_raw)

    valid_s2_ids = set(t2["entity_id"])
    valid_s3_ids = set(t3["entity_id"])
    all_s1_ids = t1["entity_id"].tolist()
    print(f"  Test data normalized: S1={len(t1):,} S2={len(t2):,} S3={len(t3):,} ({time.time()-t0:.1f}s)")

    t0 = time.time()
    # NOTE: fits a fresh vectorizer on the TEST text (not the train one) --
    # TF-IDF blocking is only used for candidate generation, not as a model
    # feature, so there's no requirement to reuse the training vocabulary,
    # and refitting here avoids losing recall on test-only vocabulary
    # (e.g. French business names/addresses that never appeared in training).
    cand2_t, cand3_t = _blocking_stage(t1, t2, t3, stage_name="Test")

    cand2_t = cand2_t[cand2_t["other_entity_id"].isin(valid_s2_ids)]
    cand3_t = cand3_t[cand3_t["other_entity_id"].isin(valid_s3_ids)]
    print(f"  Test candidates generated ({time.time()-t0:.1f}s). "
          f"cand2={len(cand2_t):,} cand3={len(cand3_t):,}")

    t0 = time.time()
    feat2_t = build_pair_features(cand2_t, t1, t2)
    feat3_t = build_pair_features(cand3_t, t1, t3)

    for col in feat_cols:
        if col not in feat2_t.columns:
            feat2_t[col] = 0.0
        if col not in feat3_t.columns:
            feat3_t[col] = 0.0

    feat2_t["score"] = booster.predict(feat2_t[feat_cols])
    feat3_t["score"] = booster.predict(feat3_t[feat_cols])
    all_feat_t = pd.concat([feat2_t, feat3_t], ignore_index=True)
    print(f"  Test features & scoring done ({time.time()-t0:.1f}s)")

    candidate_map = all_feat_t.groupby("source1_entity_id")["other_entity_id"].apply(list).to_dict()
    matched = all_feat_t[all_feat_t["score"] >= best_t]
    matched_map = matched.groupby("source1_entity_id")["other_entity_id"].apply(list).to_dict()

    sub_dir = os.path.join(out_dir, "submission")
    os.makedirs(sub_dir, exist_ok=True)
    _write_tsv(os.path.join(sub_dir, "candidate_pairs.tsv"), all_s1_ids, candidate_map, "candidate_entity_ids")
    _write_tsv(os.path.join(sub_dir, "matching_results.tsv"), all_s1_ids, matched_map, "matched_entity_ids")

    avg_matches = matched.groupby("source1_entity_id").size().mean() if len(matched) else 0.0
    print(f"  Output written to {sub_dir}. Matches found: {len(matched):,}")
    print(f"  Avg matches per S1 entity: {avg_matches:.2f} "
          f"(training ground truth average is 3.46 -- if this is wildly "
          f"different, re-check the threshold before you submit)")
    print(f"\nALL DONE in {(time.time()-t0_total)/60:.1f} min!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="../student_resource/dataset")
    parser.add_argument("--out-dir", default="./output_fast")
    args = parser.parse_args()
    run_pipeline(args.data_dir, args.out_dir)
