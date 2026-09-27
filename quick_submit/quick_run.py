"""
quick_run.py — Fast pipeline runner for Amazon ML Challenge
===========================================================

Three speed modes:

  python quick_run.py --data-dir ../dataset --out-dir ./output
      → Token blocking only, ~1–1.5 hours total. DEFAULT. Good first submission.

  python quick_run.py --data-dir ../dataset --out-dir ./output --embeddings
      → Adds name+address ANN (combined only, no separate-name ANN).
        ~4–6 hours total. Better recall on Devanagari / heavy-typo cases.

  python quick_run.py --data-dir ../dataset --out-dir ./output --full
      → Full pipeline: token + combined ANN + separate-name ANN.
        ~7–10 hours total. Max score, but only worth it if time allows.

Key speed-ups vs. the main pipeline:
  • Stratified-sampled training data (S1 cap: 80k, S2/S3 cap: 200k each).
    The model quality barely drops — LightGBM is data-efficient, and the
    important signal (what hard negatives look like) is well-represented
    even in 80k S1 rows. Inference always runs on the full test set.
  • ANN batch size bumped to 1024 for better CPU throughput.
  • Tighter candidate caps (lex=30, ann=15) reduce feature-engineering cost.
  • Blocking safety threshold raised to 5k pairs (down from 8k) for speed.
  • Checkpoints after every stage — resume from crash with same command.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import pickle
import time

# ── Make sure the parent directory (where the pipeline modules live) is on path
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)

import lightgbm as lgb
import numpy as np
import pandas as pd

from normalize import normalize_dataframe
from blocking import (
    generate_token_candidates, generate_embedding_candidates,
    cap_candidates_fair, cap_candidates_per_entity,
)
from features import build_pair_features, FEATURE_COLUMNS, quick_score
from pairs_builder import build_labeled_pairs, group_split_entities
from metrics import macro_f05, best_threshold_for_f05, candidate_recall


# ─────────────────────────────────────────────────────────────────────────────
# Speed-optimized settings  (override main config.py)
# ─────────────────────────────────────────────────────────────────────────────

# Training-data subsample caps.  Set to None to use the full dataset.
S1_TRAIN_CAP   = 120_000   # rows sampled from train_source1.tsv  (was 80k, +50% coverage)
S2_TRAIN_CAP   = 200_000   # rows sampled from train_source2.tsv
S3_TRAIN_CAP   = 200_000   # rows sampled from train_source3.tsv
GT_TRAIN_CAP   = None      # use all ground-truth rows (small file)

# Candidate caps (lower = faster feature engineering)
# Raised from 30/15 → 50/25 for better recall (more true matches reach the model)
CAP_LEXICAL    = 50
CAP_ANN        = 25

# Token-blocking safety limit — raised from 5k back to 8k (config default)
# The 5k was overly aggressive: it skips blocks with ~70 S1 x ~70 other entities,
# which is well within memory limits. The extra blocks can contain true matches.
MAX_BLOCK_PAIRS = 8_000

# ANN embedding batch size (larger = better CPU throughput)
ANN_BATCH_SIZE  = 1_024

# Workers
import config as _cfg
NUM_WORKERS = _cfg.NUM_WORKERS          # inherits from config.py (16 on r5.4xlarge)
RANDOM_SEED = _cfg.RANDOM_SEED


# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint helpers
# ─────────────────────────────────────────────────────────────────────────────

def _ckpt(ckpt_dir: str, name: str) -> str:
    return os.path.join(ckpt_dir, name)

def _save(ckpt_dir: str, name: str, data) -> None:
    os.makedirs(ckpt_dir, exist_ok=True)
    path = _ckpt(ckpt_dir, name)
    with open(path, "wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    mb = os.path.getsize(path) / 1024 / 1024
    print(f"  [ckpt] saved '{name}'  ({mb:.0f} MB)")

def _load(ckpt_dir: str, name: str):
    path = _ckpt(ckpt_dir, name)
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        data = pickle.load(f)
    mb = os.path.getsize(path) / 1024 / 1024
    print(f"  [ckpt] loaded '{name}'  ({mb:.0f} MB)")
    return data

def _has(ckpt_dir: str, name: str) -> bool:
    return os.path.exists(_ckpt(ckpt_dir, name))

def _clear(ckpt_dir: str) -> None:
    if os.path.exists(ckpt_dir):
        for f in os.listdir(ckpt_dir):
            if f.endswith(".pkl"):
                os.remove(os.path.join(ckpt_dir, f))
    print("  [ckpt] cleared checkpoints")


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _load_tsv(path: str, nrows=None) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t", nrows=nrows)

def _subsample(df: pd.DataFrame, n: int | None, name: str, seed=RANDOM_SEED) -> pd.DataFrame:
    if n is None or len(df) <= n:
        print(f"  {name}: {len(df):,} rows  (full)")
        return df
    sampled = df.sample(n=n, random_state=seed).reset_index(drop=True)
    print(f"  {name}: {len(sampled):,} rows  (sampled from {len(df):,})")
    return sampled

def _build_gt_map(gt: pd.DataFrame) -> dict:
    gt_map = {}
    for sid, ids in zip(gt["source1_entity_id"], gt["matched_entity_ids"]):
        ids = str(ids).strip()
        gt_map[sid] = set(ids.split(",")) if ids else set()
    return gt_map

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


# ─────────────────────────────────────────────────────────────────────────────
# TRAIN phase
# ─────────────────────────────────────────────────────────────────────────────

def run_train(args, ckpt_dir: str, use_ann: bool, use_name_ann: bool) -> None:
    t0_total = time.time()
    train_dir = os.path.join(args.data_dir, "train")

    # ── STAGE 1: Load + subsample + normalize ──────────────────────────────
    print("\n" + "=" * 60)
    print("STAGE 1 — Load, subsample, normalize")
    print("=" * 60)

    if _has(ckpt_dir, "train_normalized.pkl"):
        d = _load(ckpt_dir, "train_normalized.pkl")
        s1, s2, s3, gt, gt_map = d["s1"], d["s2"], d["s3"], d["gt"], d["gt_map"]
    else:
        t0 = time.time()
        s1_raw = _load_tsv(os.path.join(train_dir, "train_source1.tsv"))
        s2_raw = _load_tsv(os.path.join(train_dir, "train_source2.tsv"))
        s3_raw = _load_tsv(os.path.join(train_dir, "train_source3.tsv"))
        gt     = _load_tsv(os.path.join(train_dir, "train_ground_truth.tsv"))
        gt["matched_entity_ids"] = gt["matched_entity_ids"].fillna("")

        # Subsample — keeps entities that appear in the ground truth
        # so positives aren't wiped out.
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

        # Keep ALL gt-referenced rows, fill remaining cap from random sample
        def _smart_sample(df, id_col, must_keep_ids, cap, seed):
            if cap is None or len(df) <= cap:
                return df
            must = df[df[id_col].isin(must_keep_ids)]
            rest = df[~df[id_col].isin(must_keep_ids)]
            remaining = max(0, cap - len(must))
            filler = rest.sample(n=min(remaining, len(rest)), random_state=seed)
            return pd.concat([must, filler], ignore_index=True)

        s1_raw = _smart_sample(s1_raw, "entity_id", gt_s1_ids, S1_TRAIN_CAP, RANDOM_SEED)
        s2_raw = _smart_sample(s2_raw, "entity_id", gt_s2_ids, S2_TRAIN_CAP, RANDOM_SEED)
        s3_raw = _smart_sample(s3_raw, "entity_id", gt_s3_ids, S3_TRAIN_CAP, RANDOM_SEED)
        print(f"  Subsampled — S1={len(s1_raw):,} S2={len(s2_raw):,} S3={len(s3_raw):,}")

        # Filter GT to only S1 rows in our sample
        sampled_s1_ids = set(s1_raw["entity_id"])
        gt = gt[gt["source1_entity_id"].isin(sampled_s1_ids)].reset_index(drop=True)
        gt_map = _build_gt_map(gt)
        print(f"  GT entries after filter: {len(gt):,}")

        print("  Normalizing...")
        s1 = normalize_dataframe(s1_raw)
        s2 = normalize_dataframe(s2_raw)
        s3 = normalize_dataframe(s3_raw)
        print(f"  Normalize done  ({time.time()-t0:.0f}s)")

        _save(ckpt_dir, "train_normalized.pkl",
              {"s1": s1, "s2": s2, "s3": s3, "gt": gt, "gt_map": gt_map})

    print(f"  S1={len(s1):,}  S2={len(s2):,}  S3={len(s3):,}  GT={len(gt):,}")

    # ── STAGE 2: Blocking ──────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STAGE 2 — Blocking (token" + (" + ANN" if use_ann else "") + ")")
    print("=" * 60)

    if _has(ckpt_dir, "train_candidates.pkl"):
        d = _load(ckpt_dir, "train_candidates.pkl")
        cand2, cand3 = d["cand2"], d["cand3"]
    else:
        t0 = time.time()

        from config import NAME_STOPWORDS
        import blocking as _blk

        # Temporarily override blocking safety threshold
        _orig = _blk.MAX_TOKEN_BLOCK_PAIRS if hasattr(_blk, "MAX_TOKEN_BLOCK_PAIRS") else None
        import config as _c
        _c.MAX_TOKEN_BLOCK_PAIRS = MAX_BLOCK_PAIRS

        print("  Token blocking S1 x S2...")
        tok2 = generate_token_candidates(s1, s2)
        print("  Token blocking S1 x S3...")
        tok3 = generate_token_candidates(s1, s3)

        # Restore
        if _orig is not None:
            _c.MAX_TOKEN_BLOCK_PAIRS = _orig

        ann2 = pd.DataFrame(columns=["source1_entity_id", "other_entity_id", "ann_score"])
        ann3 = pd.DataFrame(columns=["source1_entity_id", "other_entity_id", "ann_score"])

        if use_ann:
            print("  ANN embedding S1 x S2...")
            ann2 = generate_embedding_candidates(s1, s2, batch_size=ANN_BATCH_SIZE,
                                                  use_name_ann=use_name_ann,
                                                  use_addr_ann=False)
            print("  ANN embedding S1 x S3...")
            ann3 = generate_embedding_candidates(s1, s3, batch_size=ANN_BATCH_SIZE,
                                                  use_name_ann=use_name_ann,
                                                  use_addr_ann=False)

        # Score and cap
        if not tok2.empty:
            tok2 = tok2.assign(_q=quick_score(tok2, s1, s2))
        if not tok3.empty:
            tok3 = tok3.assign(_q=quick_score(tok3, s1, s3))

        cand2 = cap_candidates_fair(tok2, ann2, "_q", max_lex=CAP_LEXICAL, max_ann=CAP_ANN)
        cand3 = cap_candidates_fair(tok3, ann3, "_q", max_lex=CAP_LEXICAL, max_ann=CAP_ANN)

        for c in ["_q"]:
            if c in cand2.columns: cand2 = cand2.drop(columns=c)
            if c in cand3.columns: cand3 = cand3.drop(columns=c)

        print(f"  Blocking done ({time.time()-t0:.0f}s)  cand2={len(cand2):,}  cand3={len(cand3):,}")

        # Blocking recall diagnostic
        all_cand = pd.concat([cand2, cand3], ignore_index=True)
        cr = candidate_recall(all_cand, gt_map)
        print(f"  Blocking recall: {cr['candidate_recall']:.4f}  "
              f"({cr['found_matches']}/{cr['total_true_matches']} true matches found)")

        _save(ckpt_dir, "train_candidates.pkl", {"cand2": cand2, "cand3": cand3})

    # ── STAGE 3: Features ──────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STAGE 3 — Feature engineering")
    print("=" * 60)

    if _has(ckpt_dir, "train_features.pkl"):
        d = _load(ckpt_dir, "train_features.pkl")
        feat2, feat3 = d["feat2"], d["feat3"]
    else:
        t0 = time.time()
        feat2 = build_pair_features(cand2, s1, s2)
        feat3 = build_pair_features(cand3, s1, s3)
        print(f"  Feature engineering done  ({time.time()-t0:.0f}s)")
        _save(ckpt_dir, "train_features.pkl", {"feat2": feat2, "feat3": feat3})

    # ── STAGE 4: Label + split ─────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STAGE 4 — Labeling + train/val split")
    print("=" * 60)

    if _has(ckpt_dir, "train_labeled.pkl"):
        d = _load(ckpt_dir, "train_labeled.pkl")
        train_df, val_df, feat_cols, val_ids = \
            d["train_df"], d["val_df"], d["feat_cols"], d["val_ids"]
    else:
        lab2 = build_labeled_pairs(feat2, gt)
        lab3 = build_labeled_pairs(feat3, gt)
        labeled = pd.concat([lab2, lab3], ignore_index=True)
        n_pos = int(labeled["label"].sum())
        print(f"  Pairs={len(labeled):,}  pos={n_pos:,}  neg={len(labeled)-n_pos:,}")

        train_ids, val_ids = group_split_entities(labeled["source1_entity_id"], 0.15, RANDOM_SEED)
        train_df = labeled[labeled["source1_entity_id"].isin(train_ids)]
        val_df   = labeled[labeled["source1_entity_id"].isin(val_ids)]
        print(f"  Train={len(train_df):,}  Val={len(val_df):,}")

        feat_cols = [c for c in FEATURE_COLUMNS if c in labeled.columns]
        if "ann_score" in labeled.columns and "ann_score" not in feat_cols:
            feat_cols.append("ann_score")

        _save(ckpt_dir, "train_labeled.pkl",
              {"train_df": train_df, "val_df": val_df,
               "feat_cols": feat_cols, "val_ids": val_ids})

    # ── STAGE 5: LightGBM ─────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STAGE 5 — LightGBM training")
    print("=" * 60)

    model_path = os.path.join(args.out_dir, "model.txt")
    if os.path.exists(model_path):
        print("  Loading existing model...")
        booster = lgb.Booster(model_file=model_path)
    else:
        dtrain = lgb.Dataset(train_df[feat_cols], label=train_df["label"])
        dval   = lgb.Dataset(val_df[feat_cols],   label=val_df["label"], reference=dtrain)
        params = {
            "objective":          "binary",
            "metric":             "auc",
            "learning_rate":      0.05,
            "num_leaves":         127,   # more expressive: 63→127
            "min_data_in_leaf":   20,
            "feature_fraction":   0.85,
            "bagging_fraction":   0.85,
            "bagging_freq":       5,
            "lambda_l1":          0.1,   # mild regularization
            "lambda_l2":          0.1,
            "is_unbalance":       True,
            "seed":               RANDOM_SEED,
            "verbosity":          -1,
            "num_threads":        NUM_WORKERS,
        }
        print(f"  Training with {len(feat_cols)} features, {NUM_WORKERS} threads...")
        booster = lgb.train(
            params, dtrain, num_boost_round=2000,
            valid_sets=[dval],
            callbacks=[lgb.early_stopping(75), lgb.log_evaluation(100)],
        )
        booster.save_model(model_path)
        print(f"  Model saved → {model_path}")

    # ── STAGE 6: Threshold tuning ──────────────────────────────────────────
    print("\n" + "=" * 60)
    print("STAGE 6 — Threshold tuning")
    print("=" * 60)

    threshold_path = os.path.join(args.out_dir, "threshold.json")
    if os.path.exists(threshold_path):
        with open(threshold_path) as f:
            meta = json.load(f)
        best_t, best_f = meta["threshold"], meta["val_f05"]
        print(f"  Loaded threshold={best_t:.4f}  val F0.5={best_f:.4f}")
    else:
        val_cand = pd.concat([
            feat2[feat2["source1_entity_id"].isin(val_ids)],
            feat3[feat3["source1_entity_id"].isin(val_ids)],
        ], ignore_index=True)
        val_cand["score"] = booster.predict(val_cand[feat_cols])
        best_t, best_f, curve = best_threshold_for_f05(val_cand, gt_map, val_ids)
        print(f"  Best threshold={best_t:.4f}  val macro F0.5={best_f:.4f}")

        with open(threshold_path, "w") as f:
            json.dump({"threshold": best_t, "val_f05": best_f,
                       "curve": curve, "feature_cols": feat_cols}, f, indent=2)

    elapsed = time.time() - t0_total
    print(f"\n  Training complete in {elapsed/60:.1f} min")


# ─────────────────────────────────────────────────────────────────────────────
# INFER phase (FULL test set — no subsampling)
# ─────────────────────────────────────────────────────────────────────────────

def run_infer(args, ckpt_dir_infer: str, use_ann: bool, use_name_ann: bool) -> None:
    t0_total = time.time()
    test_dir = os.path.join(args.data_dir, "test")

    model_path     = os.path.join(args.out_dir, "model.txt")
    threshold_path = os.path.join(args.out_dir, "threshold.json")

    if not os.path.exists(model_path) or not os.path.exists(threshold_path):
        print("ERROR: model.txt / threshold.json not found in --out-dir.")
        print("       Run training first (same command without any change).")
        sys.exit(1)

    with open(threshold_path) as f:
        meta = json.load(f)
    threshold = meta["threshold"]
    feat_cols = meta["feature_cols"]
    booster   = lgb.Booster(model_file=model_path)
    print(f"  Loaded model  threshold={threshold:.4f}  features={len(feat_cols)}")

    # ── Load + normalize test data ─────────────────────────────────────────
    print("\n" + "=" * 60)
    print("INFER STAGE 1 — Load + normalize test data  (full, no subsampling)")
    print("=" * 60)

    if _has(ckpt_dir_infer, "infer_normalized.pkl"):
        d = _load(ckpt_dir_infer, "infer_normalized.pkl")
        s1, s2, s3 = d["s1"], d["s2"], d["s3"]
    else:
        t0 = time.time()
        s1 = normalize_dataframe(_load_tsv(os.path.join(test_dir, "test_source1.tsv")))
        s2 = normalize_dataframe(_load_tsv(os.path.join(test_dir, "test_source2.tsv")))
        s3 = normalize_dataframe(_load_tsv(os.path.join(test_dir, "test_source3.tsv")))
        print(f"  Normalize done  ({time.time()-t0:.0f}s)")
        _save(ckpt_dir_infer, "infer_normalized.pkl", {"s1": s1, "s2": s2, "s3": s3})

    valid_s2_ids = set(s2["entity_id"])
    valid_s3_ids = set(s3["entity_id"])
    all_s1_ids   = s1["entity_id"].tolist()
    print(f"  S1={len(s1):,}  S2={len(s2):,}  S3={len(s3):,}")

    # ── Blocking ───────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("INFER STAGE 2 — Blocking")
    print("=" * 60)

    if _has(ckpt_dir_infer, "infer_candidates.pkl"):
        d = _load(ckpt_dir_infer, "infer_candidates.pkl")
        cand2, cand3 = d["cand2"], d["cand3"]
    else:
        t0 = time.time()
        tok2 = generate_token_candidates(s1, s2)
        tok3 = generate_token_candidates(s1, s3)

        ann2 = pd.DataFrame(columns=["source1_entity_id", "other_entity_id", "ann_score"])
        ann3 = pd.DataFrame(columns=["source1_entity_id", "other_entity_id", "ann_score"])

        if use_ann:
            print("  ANN S1 x S2...")
            ann2 = generate_embedding_candidates(s1, s2, batch_size=ANN_BATCH_SIZE,
                                                  use_name_ann=use_name_ann,
                                                  use_addr_ann=False)
            print("  ANN S1 x S3...")
            ann3 = generate_embedding_candidates(s1, s3, batch_size=ANN_BATCH_SIZE,
                                                  use_name_ann=use_name_ann,
                                                  use_addr_ann=False)

        if not tok2.empty:
            tok2 = tok2.assign(_q=quick_score(tok2, s1, s2))
        if not tok3.empty:
            tok3 = tok3.assign(_q=quick_score(tok3, s1, s3))

        cand2 = cap_candidates_fair(tok2, ann2, "_q", max_lex=CAP_LEXICAL, max_ann=CAP_ANN)
        cand3 = cap_candidates_fair(tok3, ann3, "_q", max_lex=CAP_LEXICAL, max_ann=CAP_ANN)

        for c in ["_q"]:
            if c in cand2.columns: cand2 = cand2.drop(columns=c)
            if c in cand3.columns: cand3 = cand3.drop(columns=c)

        cand2 = cand2[cand2["other_entity_id"].isin(valid_s2_ids)]
        cand3 = cand3[cand3["other_entity_id"].isin(valid_s3_ids)]
        print(f"  Blocking done ({time.time()-t0:.0f}s)  cand2={len(cand2):,}  cand3={len(cand3):,}")
        _save(ckpt_dir_infer, "infer_candidates.pkl", {"cand2": cand2, "cand3": cand3})

    # ── Features ───────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("INFER STAGE 3 — Feature engineering")
    print("=" * 60)

    if _has(ckpt_dir_infer, "infer_features.pkl"):
        d = _load(ckpt_dir_infer, "infer_features.pkl")
        feat2, feat3 = d["feat2"], d["feat3"]
    else:
        t0 = time.time()
        feat2 = build_pair_features(cand2, s1, s2)
        feat3 = build_pair_features(cand3, s1, s3)
        print(f"  Feature engineering done  ({time.time()-t0:.0f}s)")
        _save(ckpt_dir_infer, "infer_features.pkl", {"feat2": feat2, "feat3": feat3})

    # ── Score + write outputs ──────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("INFER STAGE 4 — Scoring + writing outputs")
    print("=" * 60)

    for col in feat_cols:
        if col not in feat2.columns: feat2[col] = 0.0
        if col not in feat3.columns: feat3[col] = 0.0

    feat2["score"] = booster.predict(feat2[feat_cols])
    feat3["score"] = booster.predict(feat3[feat_cols])
    all_feat = pd.concat([feat2, feat3], ignore_index=True)

    candidate_map = (all_feat
                     .groupby("source1_entity_id")["other_entity_id"]
                     .apply(list).to_dict())
    matched       = all_feat[all_feat["score"] >= threshold]
    matched_map   = (matched
                     .groupby("source1_entity_id")["other_entity_id"]
                     .apply(list).to_dict())

    n_matched = sum(len(v) for v in matched_map.values())
    print(f"  Total predicted matches: {n_matched:,}")

    sub_dir = os.path.join(args.out_dir, "submission")
    os.makedirs(sub_dir, exist_ok=True)

    _write_tsv(os.path.join(sub_dir, "candidate_pairs.tsv"),
               all_s1_ids, candidate_map, "candidate_entity_ids")
    _write_tsv(os.path.join(sub_dir, "matching_results.tsv"),
               all_s1_ids, matched_map, "matched_entity_ids")

    print(f"\n  Output written to: {sub_dir}/")
    print(f"    candidate_pairs.tsv")
    print(f"    matching_results.tsv")

    elapsed = time.time() - t0_total
    print(f"  Inference complete in {elapsed/60:.1f} min")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Fast pipeline runner — trains on a subsample, infers on the full test set.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Speed modes:
  default        Token blocking only.   ~1–1.5 hr train + ~1–2 hr infer
  --embeddings   + combined ANN.        ~3–4 hr train   + ~4–6 hr infer
  --full         + combined + name ANN. ~5–7 hr train   + ~7–10 hr infer
        """,
    )
    ap.add_argument("--data-dir",  required=True,
                    help="Parent dir containing train/ and test/ subdirs")
    ap.add_argument("--out-dir",   required=True,
                    help="Where to write model.txt, threshold.json, submission/")
    ap.add_argument("--embeddings", action="store_true",
                    help="Add combined-text (name+addr) ANN to blocking")
    ap.add_argument("--full",       action="store_true",
                    help="Add BOTH combined AND separate-name ANN (max score, slow)")
    ap.add_argument("--infer-only", action="store_true",
                    help="Skip training, jump straight to inference with existing model")
    ap.add_argument("--fresh",      action="store_true",
                    help="Ignore any existing checkpoints and start from scratch")
    args = ap.parse_args()

    use_ann      = args.embeddings or args.full
    use_name_ann = args.full

    os.makedirs(args.out_dir, exist_ok=True)

    ckpt_dir       = os.path.join(args.out_dir, "checkpoints", "train")
    ckpt_dir_infer = os.path.join(args.out_dir, "checkpoints", "infer")

    if args.fresh:
        _clear(ckpt_dir)
        _clear(ckpt_dir_infer)

    mode = "full (token + combined ANN + name ANN)" if args.full else \
           "embeddings (token + combined ANN)"      if args.embeddings else \
           "fast (token blocking only)"
    print("=" * 60)
    print(f"  Quick Submit Runner")
    print(f"  Mode : {mode}")
    print(f"  Data : {args.data_dir}")
    print(f"  Out  : {args.out_dir}")
    print(f"  CPUs : {NUM_WORKERS}")
    print("=" * 60)

    if not args.infer_only:
        run_train(args, ckpt_dir, use_ann, use_name_ann)

    run_infer(args, ckpt_dir_infer, use_ann, use_name_ann)

    print("\n" + "=" * 60)
    print("  ALL DONE — submit these files:")
    print(f"    {args.out_dir}/submission/candidate_pairs.tsv")
    print(f"    {args.out_dir}/submission/matching_results.tsv")
    print("=" * 60)


if __name__ == "__main__":
    main()
