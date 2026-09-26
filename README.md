# Business Entity Resolution — Pipeline

Blocking (token + embedding ANN) → pairwise feature engineering →
LightGBM classifier → threshold tuned to maximize macro F0.5, exactly as the
challenge scores it.

**Key features:**
- ⚡ **Parallel processing** across all CPU cores (8× on `ml.r5.2xlarge`)
- 🌐 **Language-robust**: handles Devanagari, French accents, Arabic, CJK, and any unknown script/country
- 💾 **Crash-resilient**: stage-based checkpointing with automatic resume
- 🎯 **Competition-optimized defaults**: ANN + token blocking + char n-grams enabled out of the box

## Architecture

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│  Normalize   │────▶│   Blocking   │────▶│  Features    │
│ (parallel)   │     │  (parallel)  │     │  (parallel)  │
│ + translit.  │     │  token + ANN │     │  rapidfuzz   │
└──────────────┘     └──────────────┘     └──────────────┘
       │                    │                    │
   checkpoint           checkpoint           checkpoint
                                                 │
                    ┌──────────────┐     ┌───────▼──────┐
                    │  Threshold   │◀────│  LightGBM    │
                    │  Tuning      │     │  (parallel)  │
                    └──────────────┘     └──────────────┘
```

## Files

| File | Purpose |
|------|---------|
| `config.py` | All constants, flags, normalization maps, parallelism settings |
| `normalize.py` | Name + address normalization, Devanagari + generic transliteration (parallel) |
| `blocking.py` | Token blocking (pandas merges) + FAISS ANN, parallel per-country (parallel) |
| `features.py` | Pairwise feature engineering with rapidfuzz (parallel) |
| `pairs_builder.py` | Positive/negative pair construction for training |
| `metrics.py` | Exact macro-F0.5 scorer + candidate recall diagnostics |
| `train.py` | End-to-end training with checkpointing |
| `infer.py` | End-to-end inference with checkpointing → output TSVs |

## Quick Start

```bash
pip install -r requirements.txt

# Train (ANN embeddings enabled by default)
python train.py \
    --data-dir dataset/train \
    --out-dir artifacts

# Run on test
python infer.py \
    --data-dir dataset/test \
    --model-dir artifacts \
    --out-dir output

# Token-blocking-only (faster, lower recall)
python train.py \
    --data-dir dataset/train \
    --out-dir artifacts \
    --no-embeddings
```

## SageMaker Usage

```bash
# On ml.r5.2xlarge (8 vCPUs, 64 GB RAM) — recommended
python train.py --data-dir /home/ec2-user/SageMaker/dataset/train --out-dir /home/ec2-user/SageMaker/artifacts
python infer.py --data-dir /home/ec2-user/SageMaker/dataset/test --model-dir /home/ec2-user/SageMaker/artifacts --out-dir /home/ec2-user/SageMaker/output
```

See [AWS_SAGEMAKER_GUIDE.md](AWS_SAGEMAKER_GUIDE.md) for instance selection, setup, and cost estimates.

## Parallel Processing

All CPU-intensive stages run in parallel automatically:

| Stage | Method | Estimated Speedup (8 vCPUs) |
|-------|--------|-----------------------------|
| Normalization | `multiprocessing.Pool` over DataFrame chunks | ~5–7× |
| Token blocking | `ProcessPoolExecutor` across countries | ~2–3× |
| Feature engineering | `joblib.Parallel` over pair chunks | ~5–7× |
| Candidate ranking (`quick_score`) | `joblib.Parallel` over pair chunks | ~5–7× |
| LightGBM training | Native `num_threads=8` | ~2–4× |

Falls back to single-process for small datasets (<1000 rows / <10k pairs).

## Checkpoint / Resume

The pipeline automatically saves checkpoints after each major stage:

### train.py checkpoints
```
artifacts/checkpoints/ckpt_normalized.pkl      (after normalization)
artifacts/checkpoints/ckpt_candidates.pkl      (after blocking + capping)
artifacts/checkpoints/ckpt_features.pkl        (after feature engineering)
artifacts/checkpoints/ckpt_labeled.pkl         (after labeling + train/val split)
artifacts/model.txt                            (after LightGBM training)
artifacts/threshold.json                       (after threshold tuning)
```

### infer.py checkpoints
```
output/checkpoints/ckpt_infer_normalized.pkl
output/checkpoints/ckpt_infer_candidates.pkl
output/checkpoints/ckpt_infer_features.pkl
```

### Resume behavior
- **Default**: Resume is enabled. If a crash occurs, re-running the same command skips completed stages.
- `--fresh`: Force restart from scratch (clears all checkpoints).
- `--checkpoint-dir PATH`: Use a custom checkpoint directory.

## Language Robustness

The pipeline handles any language/script in test data, even if absent from training:

| Script | Handling |
|--------|----------|
| Latin (English) | Full normalization + fuzzy features |
| Accented Latin (French, German, Spanish) | `unicodedata.NFKD` → stripped accents → "Société" becomes "Societe" |
| Devanagari (Hindi) | Rule-based phonetic transliteration → "कंपनी" becomes "knpni" |
| Arabic, CJK, Cyrillic, Thai | ASCII fallback (characters stripped). Embedding model (`multilingual-e5-small`) handles cross-script matching natively |
| Unknown countries | Graceful fallback: no state extraction, generic postal code regex, blocking/features still work via name tokens + embeddings |

## Competition-Optimized Defaults

All defaults are tuned for maximum F0.5:

| Setting | Value | Why |
|---------|-------|-----|
| `USE_TOKEN_BLOCKING` | `True` | Primary recall driver |
| `--use-embeddings` | `True` (default) | Catches Devanagari↔Latin, heavy typos. ~5% of Indian names are in Devanagari |
| `USE_SEPARATE_NAME_ANN` | `True` | Name-only ANN helps when addresses are very different |
| `ANN_TOP_K` | `20` | Higher K = better recall ceiling |
| `MAX_TOKEN_BLOCK_PAIRS` | `8,000` | Higher than 5K catches more dense-block matches |
| `USE_CHAR_NGRAM_FEATURES` | `True` | 3/4-gram cosine sim catches typos, abbreviations |
| `USE_ASYMMETRIC_E5_PREFIXES` | `True` | E5 model works better with query:/passage: prefixes |
| `CANDIDATE_CAP_LEXICAL` | `40` | Keeps top-40 lexical candidates per entity |
| `CANDIDATE_CAP_ANN` | `20` | Guarantees 20 ANN slots (separate pool from lexical) |

## Outputs

`train.py` produces:
- `artifacts/model.txt` — LightGBM booster
- `artifacts/threshold.json` — tuned threshold + F0.5 curve + feature column list
- `artifacts/feature_importance.csv` — feature importance ranking

`infer.py` produces (tab-separated, per spec):
- `output/candidate_pairs.tsv` — all blocking candidates (pre-threshold)
- `output/matching_results.tsv` — final predicted matches (post-threshold)

## Estimated Runtime (ml.r5.2xlarge, 8 vCPUs, 64 GB)

| Phase | Step | Estimated Time |
|-------|------|----------------|
| Train | Load + normalize | 3–8 min |
| | Token blocking | 3–10 min |
| | ANN embeddings | 30–90 min (CPU; 10× faster on GPU) |
| | Feature engineering | 5–15 min |
| | LightGBM + threshold | 3–8 min |
| **Total train** | | **~45–130 min** |
| Infer | Load + normalize | 3–8 min |
| | Blocking + features | 15–50 min |
| | Scoring | 1–3 min |
| **Total infer** | | **~20–60 min** |

> **Tip**: For the ANN stage, a GPU instance (`ml.g4dn.xlarge`) is 10× faster.
> Use `--no-embeddings` for a quick token-only run (~30 min total).

## Known Gaps

- Address normalization only has gazetteers for US and India states. Unknown countries
  fall back to generic heuristics (still works, just loses state/city feature signal).
- CJK/Arabic/Cyrillic business names are stripped to empty by the fuzzy features;
  matching relies entirely on the embedding model for these scripts.
- Consider fine-tuning the embedding model on ground-truth pairs for higher blocking recall.
- A graph-consistency post-processing pass (corroborating S2↔S3 matches) could boost precision.

## Requirements

```
pandas>=2.1, numpy>=1.26, rapidfuzz>=3.6, lightgbm>=4.3
scikit-learn>=1.4, joblib>=1.3, pyarrow>=15.0
# Optional (for ANN, enabled by default):
sentence-transformers>=2.6, faiss-cpu>=1.8, torch>=2.2
```
