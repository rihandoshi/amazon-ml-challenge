import os
import sys
import pandas as pd
import time

# Add paths
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)

from normalize import normalize_dataframe
from tfidf_blocking import tfidf_blocking
from blocking import generate_token_candidates

import config
config.NUM_WORKERS = 1

def run_test():
    print("Testing TF-IDF pipeline components...")
    
    data_dir = os.path.join(_ROOT, "student_resource", "dataset")
    train_dir = os.path.join(data_dir, "train")
    
    t0 = time.time()
    s1_raw = pd.read_csv(os.path.join(train_dir, "train_source1.tsv"), sep="\t", nrows=5000)
    s2_raw = pd.read_csv(os.path.join(train_dir, "train_source2.tsv"), sep="\t", nrows=10000)
    
    print("1. Normalizing...")
    s1 = normalize_dataframe(s1_raw, n_workers=1)
    s2 = normalize_dataframe(s2_raw, n_workers=1)
    
    print("2. TF-IDF Blocking...")
    tfidf_cand = tfidf_blocking(s1, s2, text_col='norm_name', top_k=10, min_score=0.1)
    print(f"   Generated {len(tfidf_cand)} candidates.")
    print(tfidf_cand.head())
    
    assert 'ann_score' in tfidf_cand.columns, "ann_score column missing!"
    assert not tfidf_cand.empty, "TF-IDF blocking returned empty dataframe!"
    
    print("3. Feature Generation test (mocking pairs)...")
    from features import build_pair_features
    # Combine with some token candidates just to test fair cap
    tok_cand = generate_token_candidates(s1, s2)
    from features import quick_score
    from blocking import cap_candidates_fair
    
    if not tok_cand.empty:
        tok_cand = tok_cand.assign(_q=quick_score(tok_cand, s1, s2))
        
    cand_capped = cap_candidates_fair(tok_cand, tfidf_cand, "_q", score_col_ann="ann_score", max_lex=10, max_ann=10)
    print(f"   Capped candidates: {len(cand_capped)}")
    
    feat = build_pair_features(cand_capped, s1, s2)
    print(f"   Features built! Shape: {feat.shape}")
    print("All tests passed successfully in {:.1f}s".format(time.time() - t0))

if __name__ == "__main__":
    run_test()
