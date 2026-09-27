"""Quick diagnostic: is the Devanagari-vs-Latin script gap actually costing
you recall, or is it already covered by zip/city/token blocking?

Run this on a training sample (a few minutes, not the full 2.2M rows) while
your main train_fast.py run is going. It tells you the one thing you can't
just guess: whether it's worth spending remaining time on script handling
at all.

Usage:
    python diagnose_recall.py --data-dir /path/to/dataset/train [--sample 50000]
"""
import argparse
import os
import re
import sys

import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from normalize import normalize_dataframe
from blocking import generate_token_candidates, cap_candidates_fair
from features import quick_score
from metrics import candidate_recall
from tfidf_blocking import tfidf_blocking_multi

_DEVANAGARI_RE = re.compile(r"[\u0900-\u097F]")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--sample", type=int, default=50_000,
                     help="rows to sample from S2/S3 (S1 is filtered to entities with GT + a sample of singletons)")
    ap.add_argument("--top-k", type=int, default=30)
    ap.add_argument("--min-score", type=float, default=0.2)
    args = ap.parse_args()

    print("Loading...")
    gt = pd.read_csv(os.path.join(args.data_dir, "train_ground_truth.tsv"), sep="\t")
    gt["matched_entity_ids"] = gt["matched_entity_ids"].fillna("")
    s1_full = pd.read_csv(os.path.join(args.data_dir, "train_source1.tsv"), sep="\t")
    s2_full = pd.read_csv(os.path.join(args.data_dir, "train_source2.tsv"), sep="\t")
    s3_full = pd.read_csv(os.path.join(args.data_dir, "train_source3.tsv"), sep="\t")

    gt_pos = gt[gt["matched_entity_ids"] != ""]
    s1_ids = set(gt_pos["source1_entity_id"].sample(
        min(args.sample // 10, len(gt_pos)), random_state=42))

    needed_s2, needed_s3 = set(), set()
    for ids in gt_pos[gt_pos["source1_entity_id"].isin(s1_ids)]["matched_entity_ids"]:
        for i in str(ids).split(","):
            if i.startswith("S2-"):
                needed_s2.add(i)
            elif i.startswith("S3-"):
                needed_s3.add(i)

    s1 = s1_full[s1_full["entity_id"].isin(s1_ids)]
    s2 = pd.concat([
        s2_full[s2_full["entity_id"].isin(needed_s2)],
        s2_full.sample(min(args.sample, len(s2_full)), random_state=1),
    ]).drop_duplicates("entity_id")
    s3 = pd.concat([
        s3_full[s3_full["entity_id"].isin(needed_s3)],
        s3_full.sample(min(args.sample, len(s3_full)), random_state=2),
    ]).drop_duplicates("entity_id")
    gt_small = gt[gt["source1_entity_id"].isin(s1_ids)]
    gt_map = {sid: set(ids.split(",")) if ids else set()
              for sid, ids in zip(gt_small["source1_entity_id"], gt_small["matched_entity_ids"])}

    print(f"Sample sizes: S1={len(s1)} S2={len(s2)} S3={len(s3)}")

    print("Normalizing...")
    s1n = normalize_dataframe(s1)
    s2n = normalize_dataframe(s2)
    s3n = normalize_dataframe(s3)

    print("Blocking (token + TF-IDF name)...")
    tok2 = generate_token_candidates(s1n, s2n)
    tok3 = generate_token_candidates(s1n, s3n)
    name_results = tfidf_blocking_multi(s1n, {"s2": s2n, "s3": s3n}, text_col="norm_name",
                                         top_k=args.top_k, min_score=args.min_score)
    if not tok2.empty:
        tok2 = tok2.assign(_q=quick_score(tok2, s1n, s2n))
    if not tok3.empty:
        tok3 = tok3.assign(_q=quick_score(tok3, s1n, s3n))
    cand2 = cap_candidates_fair(tok2, name_results["s2"], "_q", score_col_ann="ann_score", max_lex=40, max_ann=30)
    cand3 = cap_candidates_fair(tok3, name_results["s3"], "_q", score_col_ann="ann_score", max_lex=40, max_ann=30)
    all_cand = pd.concat([cand2, cand3], ignore_index=True)

    overall = candidate_recall(all_cand, gt_map)
    print(f"\nOverall candidate recall: {overall['candidate_recall']:.4f} "
          f"({overall.get('found', '?')}/{overall.get('total', '?')} true matches recovered)")

    # Which ground-truth matches are Devanagari-script on the OTHER side?
    dev_ids = set(s2n.loc[s2n["business_name"].apply(
        lambda x: bool(_DEVANAGARI_RE.search(str(x)))), "entity_id"])
    dev_ids |= set(s3n.loc[s3n["business_name"].apply(
        lambda x: bool(_DEVANAGARI_RE.search(str(x)))), "entity_id"])

    dev_gt_map = {}
    non_dev_gt_map = {}
    for sid, ids in gt_map.items():
        dev_part = {i for i in ids if i in dev_ids}
        non_dev_part = ids - dev_part
        if dev_part:
            dev_gt_map[sid] = dev_part
        if non_dev_part:
            non_dev_gt_map[sid] = non_dev_part

    if dev_gt_map:
        dev_recall = candidate_recall(all_cand, dev_gt_map)
        print(f"Recall on Devanagari-script true matches only: "
              f"{dev_recall['candidate_recall']:.4f} "
              f"({len(dev_gt_map)} S1 entities affected in this sample)")
    else:
        print("No Devanagari-script true matches landed in this sample -- "
              "re-run with a bigger --sample if you want this number.")

    if non_dev_gt_map:
        non_dev_recall = candidate_recall(all_cand, non_dev_gt_map)
        print(f"Recall on non-Devanagari true matches: {non_dev_recall['candidate_recall']:.4f}")

    print("\nInterpretation: if the Devanagari recall is within ~2-3 points of "
          "the non-Devanagari recall, your zip/city/token blocking is already "
          "covering the script gap and you don't need to spend more time on it "
          "tonight. If it's substantially lower, that's your biggest remaining "
          "lever -- but with the clock you have, shipping what you've got is "
          "still the right call over building script handling from scratch.")


if __name__ == "__main__":
    main()
