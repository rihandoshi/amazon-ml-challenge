"""Quick sanity test: load a small subset of the real data, run through
the full pipeline (normalize -> block -> features -> label), and verify
no crashes, no memory explosions, and schemas are correct.

Usage:
    .venv/Scripts/python.exe tests/test_pipeline_sanity.py
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         "student_resource", "dataset", "train")

def load_subset(name, n=500):
    path = os.path.join(DATA_DIR, name)
    if not os.path.exists(path):
        print(f"  SKIP: {path} not found")
        return None
    return pd.read_csv(path, sep="\t", nrows=n)


def main():
    print("=" * 60)
    print("Pipeline sanity test (small subset)")
    print("=" * 60)

    # Load small subsets
    s1_raw = load_subset("train_source1.tsv", 200)
    s2_raw = load_subset("train_source2.tsv", 500)
    s3_raw = load_subset("train_source3.tsv", 500)
    gt_raw = load_subset("train_ground_truth.tsv", 5000)

    if any(x is None for x in [s1_raw, s2_raw, s3_raw, gt_raw]):
        print("Some data files missing, skipping test")
        return

    print(f"\nSubset sizes: S1={len(s1_raw)} S2={len(s2_raw)} S3={len(s3_raw)} GT={len(gt_raw)}")

    # Step 1: Normalize
    print("\n--- Normalization ---")
    from normalize import normalize_dataframe
    s1 = normalize_dataframe(s1_raw)
    s2 = normalize_dataframe(s2_raw)
    s3 = normalize_dataframe(s3_raw)

    required_cols = ["entity_id", "norm_name", "name_tokens", "norm_addr",
                     "zip", "state", "city", "house_no", "country", "has_devanagari"]
    for col in required_cols:
        assert col in s1.columns, f"Missing column {col} in normalized S1"
    print(f"  S1 columns: {list(s1.columns)}")
    print(f"  Sample norm_name: {s1['norm_name'].iloc[0]}")
    print(f"  Sample city: {s1['city'].iloc[0]}")
    print("  PASS: Normalization schema correct")

    # Step 2: Token blocking (with frequency-aware safety)
    print("\n--- Token Blocking ---")
    from blocking import generate_token_candidates
    cand2 = generate_token_candidates(s1, s2)
    cand3 = generate_token_candidates(s1, s3)
    assert "source1_entity_id" in cand2.columns
    assert "other_entity_id" in cand2.columns
    print(f"  Token candidates: S2={len(cand2)} S3={len(cand3)}")
    print("  PASS: Token blocking works (no memory explosion)")

    # Step 3: Fair candidate capping
    print("\n--- Fair Candidate Capping ---")
    from blocking import cap_candidates_fair
    from features import quick_score

    # Create fake ANN candidates for testing
    ann2 = pd.DataFrame({
        "source1_entity_id": cand2["source1_entity_id"].iloc[:min(20, len(cand2))].values,
        "other_entity_id": cand2["other_entity_id"].iloc[:min(20, len(cand2))].values,
        "ann_score": np.random.rand(min(20, len(cand2))),
    }) if len(cand2) > 0 else pd.DataFrame(columns=["source1_entity_id", "other_entity_id", "ann_score"])

    if not cand2.empty:
        cand2_scored = cand2.assign(_q=quick_score(cand2, s1, s2))
    else:
        cand2_scored = cand2

    capped = cap_candidates_fair(cand2_scored, ann2, score_col_lex="_q")
    assert "source1_entity_id" in capped.columns
    assert "other_entity_id" in capped.columns
    print(f"  Capped candidates: {len(capped)}")
    if "from_token_block" in capped.columns:
        print(f"  With from_token_block: {capped['from_token_block'].sum()}")
    if "from_ann" in capped.columns:
        print(f"  With from_ann: {capped['from_ann'].sum()}")
    print("  PASS: Fair capping works")

    # Step 4: Feature engineering (including char n-grams)
    print("\n--- Feature Engineering ---")
    from features import build_pair_features, FEATURE_COLUMNS

    if not cand2.empty:
        feats = build_pair_features(cand2.head(50), s1, s2)
        print(f"  Feature columns: {list(feats.columns)}")
        print(f"  Feature rows: {len(feats)}")

        # Check char n-gram features are present
        for col in ["name_char_3gram_cosine", "name_char_4gram_cosine", "addr_char_3gram_cosine"]:
            assert col in feats.columns, f"Missing feature {col}"
        print(f"  name_char_3gram range: [{feats['name_char_3gram_cosine'].min():.3f}, {feats['name_char_3gram_cosine'].max():.3f}]")
        print(f"  name_char_4gram range: [{feats['name_char_4gram_cosine'].min():.3f}, {feats['name_char_4gram_cosine'].max():.3f}]")
        print("  PASS: Features including char n-grams work")
    else:
        print("  SKIP: no candidates to featurize")

    # Step 5: Candidate recall
    print("\n--- Candidate Recall ---")
    from metrics import candidate_recall

    gt_raw["matched_entity_ids"] = gt_raw["matched_entity_ids"].fillna("")
    gt_map = {}
    for sid, ids in zip(gt_raw["source1_entity_id"], gt_raw["matched_entity_ids"]):
        ids = str(ids).strip()
        gt_map[sid] = set(ids.split(",")) if ids else set()

    all_cand = pd.concat([cand2, cand3], ignore_index=True)
    cr = candidate_recall(all_cand, gt_map)
    print(f"  Candidate recall: {cr['candidate_recall']:.4f}")
    print(f"  Found {cr['found_matches']} / {cr['total_true_matches']} true matches")
    print("  PASS: Candidate recall metric works")

    # Step 6: Labeling
    print("\n--- Labeling ---")
    from pairs_builder import build_labeled_pairs

    if not cand2.empty:
        feats = build_pair_features(cand2, s1, s2)
        labeled = build_labeled_pairs(feats, gt_raw)
        print(f"  Labeled pairs: {len(labeled)}")
        print(f"  Positives: {labeled['label'].sum()}")
        print(f"  Negatives: {(labeled['label'] == 0).sum()}")
        print("  PASS: Labeling works")
    else:
        print("  SKIP: no candidates to label")

    print("\n" + "=" * 60)
    print("ALL SANITY CHECKS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()
