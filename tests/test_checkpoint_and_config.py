"""Quick test: checkpoint save/load roundtrip."""
import sys, os, pickle, tempfile, shutil
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

# Simulate checkpoint operations
ckpt_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         "tests", "_test_ckpts")
os.makedirs(ckpt_dir, exist_ok=True)

try:
    # Test 1: Save and load a checkpoint
    print("=== Checkpoint Save/Load Test ===")
    test_data = {
        "df": pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]}),
        "scalar": 42,
        "dict": {"key": "value"},
    }
    path = os.path.join(ckpt_dir, "test_ckpt.pkl")
    with open(path, "wb") as f:
        pickle.dump(test_data, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    with open(path, "rb") as f:
        loaded = pickle.load(f)
    
    assert loaded["scalar"] == 42
    assert len(loaded["df"]) == 3
    assert loaded["dict"]["key"] == "value"
    print("  PASS: Checkpoint roundtrip works")

    # Test 2: Verify config values
    print("\n=== Config Defaults Test ===")
    from config import (
        NUM_WORKERS, CHECKPOINT_ENABLED, USE_TOKEN_BLOCKING,
        USE_SEPARATE_NAME_ANN, ANN_TOP_K, MAX_TOKEN_BLOCK_PAIRS,
        USE_CHAR_NGRAM_FEATURES, USE_ASYMMETRIC_E5_PREFIXES,
    )
    print(f"  NUM_WORKERS: {NUM_WORKERS}")
    print(f"  CHECKPOINT_ENABLED: {CHECKPOINT_ENABLED}")
    print(f"  USE_TOKEN_BLOCKING: {USE_TOKEN_BLOCKING}")
    print(f"  USE_SEPARATE_NAME_ANN: {USE_SEPARATE_NAME_ANN}")
    print(f"  ANN_TOP_K: {ANN_TOP_K}")
    print(f"  MAX_TOKEN_BLOCK_PAIRS: {MAX_TOKEN_BLOCK_PAIRS}")
    print(f"  USE_CHAR_NGRAM_FEATURES: {USE_CHAR_NGRAM_FEATURES}")
    print(f"  USE_ASYMMETRIC_E5_PREFIXES: {USE_ASYMMETRIC_E5_PREFIXES}")
    
    assert CHECKPOINT_ENABLED == True
    assert USE_TOKEN_BLOCKING == True
    assert USE_SEPARATE_NAME_ANN == True
    assert ANN_TOP_K == 20
    assert MAX_TOKEN_BLOCK_PAIRS == 8_000
    assert USE_CHAR_NGRAM_FEATURES == True
    assert NUM_WORKERS >= 1
    print("  PASS: All defaults are competition-optimal")

    # Test 3: Verify parallel features work
    print("\n=== Parallel Feature Test ===")
    from normalize import normalize_dataframe
    from features import build_pair_features, quick_score

    DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                             "student_resource", "dataset", "train")
    s1_raw = pd.read_csv(os.path.join(DATA_DIR, "train_source1.tsv"), sep="\t", nrows=100)
    s2_raw = pd.read_csv(os.path.join(DATA_DIR, "train_source2.tsv"), sep="\t", nrows=200)
    
    s1 = normalize_dataframe(s1_raw)
    s2 = normalize_dataframe(s2_raw)
    
    from blocking import generate_token_candidates
    cands = generate_token_candidates(s1, s2)
    
    if not cands.empty:
        feats = build_pair_features(cands, s1, s2)
        scores = quick_score(cands, s1, s2)
        print(f"  Pairs: {len(cands):,}, Features: {len(feats.columns)}, Scores: {len(scores)}")
        print("  PASS: Parallel features + quick_score work")
    else:
        print("  SKIP: No candidates generated (small sample)")

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED ✓")
    print("=" * 60)

finally:
    # Cleanup
    if os.path.exists(ckpt_dir):
        shutil.rmtree(ckpt_dir)
