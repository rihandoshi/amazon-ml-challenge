"""Run this right before you upload matching_results.tsv.

Checks the average matches-per-S1-entity against the training ground truth's
known average (3.46) and flags a likely threshold miscalibration. This is
NOT a replacement for the organizers' utils/validate_submission.py (still
run that too) -- this just catches "your threshold is way off" fast, which
validate_submission.py can't tell you since it only checks format.

Usage:
    python check_submission.py --matching output/submission/matching_results.tsv \
        --candidate output/submission/candidate_pairs.tsv \
        --train-gt dataset/train/train_ground_truth.tsv
"""
import argparse
import pandas as pd


def load_id_lists(path, col):
    df = pd.read_csv(path, sep="\t", keep_default_na=False)
    lens = df[col].apply(lambda s: 0 if s == "" else len(s.split(",")))
    return df, lens


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--matching", required=True)
    ap.add_argument("--candidate", required=True)
    ap.add_argument("--train-gt", required=False,
                     help="train_ground_truth.tsv, to compare distributions (optional but recommended)")
    args = ap.parse_args()

    m_df, m_lens = load_id_lists(args.matching, "matched_entity_ids")
    c_df, c_lens = load_id_lists(args.candidate, "candidate_entity_ids")

    print(f"matching_results.tsv: {len(m_df):,} rows, "
          f"avg matches/entity = {m_lens.mean():.3f}, "
          f"singleton (empty) rate = {(m_lens == 0).mean():.3f}")
    print(f"candidate_pairs.tsv:  {len(c_df):,} rows, "
          f"avg candidates/entity = {c_lens.mean():.3f}")

    if args.train_gt:
        gt = pd.read_csv(args.train_gt, sep="\t")
        gt["matched_entity_ids"] = gt["matched_entity_ids"].fillna("")
        gt_lens = gt["matched_entity_ids"].apply(lambda s: 0 if s == "" else len(s.split(",")))
        print(f"\ntrain_ground_truth.tsv (reference): "
              f"avg matches/entity = {gt_lens.mean():.3f}, "
              f"singleton rate = {(gt_lens == 0).mean():.3f}")

        ratio = m_lens.mean() / gt_lens.mean() if gt_lens.mean() else float("inf")
        print(f"\nYour avg / ground-truth avg = {ratio:.2f}x")
        if ratio > 1.5:
            print("  -> WARNING: you're predicting notably MORE matches per entity than "
                  "training ground truth. Given F0.5's precision weighting, this usually "
                  "means your threshold is too low -- check threshold.json / the sweep curve "
                  "and consider raising it before submitting.")
        elif ratio < 0.6:
            print("  -> WARNING: you're predicting notably FEWER matches per entity than "
                  "training ground truth. Could be a threshold that's too high, or a "
                  "blocking recall gap. Check candidate_pairs.tsv's avg candidates/entity above -- "
                  "if that's also low, it's a blocking problem, not a threshold problem.")
        else:
            print("  -> In a reasonable range relative to training ground truth.")

    singleton_rate = (m_lens == 0).mean()
    if singleton_rate < 0.02:
        print("\nNOTE: your singleton rate is very low (training data is ~5.6% singletons). "
              "F0.5 rewards correctly predicting empty on true singletons -- if your model "
              "almost never predicts empty, that's likely costing you points on the "
              "~1-in-18 entities that truly have no match.")

    # sanity: matches should be a subset of candidates, per entity
    m_map = dict(zip(m_df["source1_entity_id"],
                      m_df["matched_entity_ids"].apply(lambda s: set(s.split(",")) if s else set())))
    c_map = dict(zip(c_df["source1_entity_id"],
                      c_df["candidate_entity_ids"].apply(lambda s: set(s.split(",")) if s else set())))
    bad = 0
    for sid, matched_ids in m_map.items():
        if not matched_ids:
            continue
        if not matched_ids.issubset(c_map.get(sid, set())):
            bad += 1
    if bad:
        print(f"\nWARNING: {bad} entities have a matched id that never appeared in "
              f"candidate_pairs.tsv -- this is exactly the pipeline-bug signal the "
              f"organizers' validator also checks for. Fix before submitting.")
    else:
        print("\nOK: every match is a subset of its candidates.")


if __name__ == "__main__":
    main()
