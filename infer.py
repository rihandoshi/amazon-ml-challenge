"""Run the full pipeline on the test set and write the two required outputs.

Usage:
    python infer.py --data-dir /path/to/dataset/test --model-dir /path/to/artifacts \
        --out-dir /path/to/output [--use-embeddings] [--threshold 0.5]

Writes, tab-separated, exactly per the spec:
    <out-dir>/candidate_pairs.tsv      (last blocking-stage candidates, pre-threshold)
    <out-dir>/matching_results.tsv     (final matches, post-threshold)

Every Source-1 test entity gets exactly one row in both files (empty string
for no candidates / no matches), matching_entity_ids only ever reference
S2-/S3- ids present in the test files, and matches are a subset of candidates.
"""
import argparse
import json
import os

import lightgbm as lgb
import pandas as pd

from normalize import normalize_dataframe
from blocking import generate_token_candidates, generate_embedding_candidates, cap_candidates_per_entity
from features import build_pair_features, quick_score


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
    ap.add_argument("--use-embeddings", action="store_true")
    ap.add_argument("--threshold", type=float, default=None, help="override the tuned threshold")
    ap.add_argument("--max-candidates-per-entity", type=int, default=50)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    print("Loading + normalizing test data...")
    s1 = normalize_dataframe(load_source(args.data_dir, "test_source1.tsv"))
    s2 = normalize_dataframe(load_source(args.data_dir, "test_source2.tsv"))
    s3 = normalize_dataframe(load_source(args.data_dir, "test_source3.tsv"))

    valid_s2_ids = set(s2["entity_id"])
    valid_s3_ids = set(s3["entity_id"])
    all_s1_ids = s1["entity_id"].tolist()

    print("Blocking...")
    cand2 = generate_token_candidates(s1, s2)
    cand3 = generate_token_candidates(s1, s3)
    if args.use_embeddings:
        ann2 = generate_embedding_candidates(s1, s2)
        cand2 = pd.concat([cand2, ann2[["source1_entity_id", "other_entity_id"]]]).drop_duplicates()
        ann3 = generate_embedding_candidates(s1, s3)
        cand3 = pd.concat([cand3, ann3[["source1_entity_id", "other_entity_id"]]]).drop_duplicates()

    if args.max_candidates_per_entity:
        cand2 = cand2.assign(_q=quick_score(cand2, s1, s2))
        cand2 = cap_candidates_per_entity(cand2, "_q", args.max_candidates_per_entity).drop(columns="_q")
        cand3 = cand3.assign(_q=quick_score(cand3, s1, s3))
        cand3 = cap_candidates_per_entity(cand3, "_q", args.max_candidates_per_entity).drop(columns="_q")

    # sanity: candidates must only reference ids that actually exist in test files
    cand2 = cand2[cand2["other_entity_id"].isin(valid_s2_ids)]
    cand3 = cand3[cand3["other_entity_id"].isin(valid_s3_ids)]

    print("Featurizing...")
    feat2 = build_pair_features(cand2, s1, s2)
    feat3 = build_pair_features(cand3, s1, s3)

    with open(os.path.join(args.model_dir, "threshold.json")) as f:
        meta = json.load(f)
    threshold = args.threshold if args.threshold is not None else meta["threshold"]
    feat_cols = meta["feature_cols"]
    print(f"Using threshold={threshold}")

    booster = lgb.Booster(model_file=os.path.join(args.model_dir, "model.txt"))
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

    write_id_list_tsv(os.path.join(args.out_dir, "candidate_pairs.tsv"), all_s1_ids, candidate_map)
    write_id_list_tsv(os.path.join(args.out_dir, "matching_results.tsv"), all_s1_ids, matched_map)
    print("Wrote", os.path.join(args.out_dir, "candidate_pairs.tsv"))
    print("Wrote", os.path.join(args.out_dir, "matching_results.tsv"))


if __name__ == "__main__":
    main()


