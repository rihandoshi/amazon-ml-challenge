# Business Entity Resolution — pipeline

Blocking (token + optional embedding ANN) → pairwise feature engineering →
LightGBM classifier → threshold tuned to maximize macro F0.5, exactly as the
challenge scores it.

This has been **smoke-tested end to end** on a ~1,000-entity sample cut from
your real training data (`train.py` → `infer.py` → a hand-rolled validator
matching the official rules) and produces correctly-formatted, zero-violation
output. It has **not** been run on the full 2.2M / 5M / 5.3M row files — see
"Scaling to the real dataset" below for what changes.

## Why this shape, not a straight XGBoost-on-everything pipeline

Your original plan (normalize → block → pairwise features → XGBoost →
threshold) is the right skeleton — this keeps it, and changes three things
that matter at this data's actual scale and noise profile:

1. **Blocking has to be the main event, not a preprocessing footnote.**
   S1 × S2 × S3 is ~2.2M × 5M × 5.3M — brute-force pairing is ~10¹³ pairs,
   completely infeasible. Everything downstream is capped by what blocking
   recovers, so it gets two independent mechanisms, unioned:
   - **Token blocking** (exact match on normalized name tokens / zip /
     city+first-token, vectorized as pandas merges, partitioned by country).
     Cheap, scales linearly, catches "same words, reordered/abbreviated".
   - **Embedding ANN** (multilingual bi-encoder + FAISS). I checked your
     actual data: **~5.3% of Indian Source-2 names and ~3.0% of Source-3
     names are in Devanagari script**, while Source-1 names are always
     Latin. Token blocking can't bridge that at all — there's no shared
     token. An embedding model that's seen both scripts can. This is also
     what recovers heavy typos / transpositions that share zero tokens.
   Cap candidates per entity (default 50) by a cheap fuzzy score before the
   expensive feature stage, so compute stays bounded regardless of how many
   candidates blocking floods a given entity with.

2. **F0.5 is macro-averaged per entity and heavily precision-weighted** — a
   false merge costs you 2× what a miss costs, and singletons (5.6% of your
   training S1 entities) score 1.0 for correctly predicting *nothing* and
   0.0 for any false positive. That means: (a) the decision threshold
   matters more than model AUC, so `train.py` sweeps thresholds on a
   held-out split and picks the one that actually maximizes macro F0.5
   (not accuracy, not F1); (b) negative sampling during training uses
   blocking's own near-misses as negatives — the classifier is trained
   specifically to reject the confusable candidates it will actually see
   at inference, not random unrelated pairs, which is what gives you
   precision on the hard cases.

3. **France in the test set, absent from training, with country treated as
   an open string.** Nothing in the pipeline hardcodes `{US, India}` —
   blocking partitions by whatever country value shows up, and the
   embedding model (multilingual, not English-only) doesn't need to have
   seen French business names to embed them sensibly. You should still
   spot-check France-like synthetic addresses once you can construct some,
   since the address-component regexes (US zip / Indian PIN / state
   gazetteers) currently only understand US and India formats — see
   "Known gaps" below.

## Files

```
src/
  config.py          constants: name/address abbreviation maps, US/India state
                      gazetteers, embedding model id
  normalize.py        name + address normalization, Devanagari transliteration
  blocking.py          token blocking (pandas merges) + optional FAISS ANN
  features.py           pairwise feature engineering (rapidfuzz-based)
  pairs_builder.py       positive/negative pair construction for training
  metrics.py              exact macro-F0.5 scorer from the problem statement
  train.py                 end-to-end training entrypoint
  infer.py                  end-to-end inference entrypoint -> the two required TSVs
```

## Running it

```bash
pip install -r requirements.txt

# 1. Train (point at the folder with train_source1/2/3.tsv + train_ground_truth.tsv)
python src/train.py \
    --data-dir dataset/train \
    --out-dir artifacts \
    --use-embeddings          # omit for a faster token-only-blocking first pass

# 2. Run on test (point at the folder with test_source1/2/3.tsv)
python src/infer.py \
    --data-dir dataset/test \
    --model-dir artifacts \
    --out-dir output \
    --use-embeddings

# 3. Validate against the organizers' own rules before you spend a submission
python3 utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir dataset/test
```

`train.py` writes `artifacts/model.txt`, `artifacts/threshold.json` (tuned
threshold + the full threshold→F0.5 curve — worth eyeballing, a flat curve
near the max means the threshold choice is robust; a sharp spike means it's
sensitive and you should cross-validate it), and
`artifacts/feature_importance.csv`.

## Scaling to the real dataset (this is the part that needs SageMaker)

The smoke test used ~1K/4K/4.5K row samples. At the real 2.2M/5M/5.3M scale:

- **Run blocking and feature engineering per-country** (already how
  `generate_token_candidates` is structured) on a high-memory instance
  (e.g. `r6i.8xlarge`/`r6i.16xlarge`) or convert the pandas merges to
  Dask/PySpark if a single country partition (US will dominate) still
  doesn't fit in memory.
- **Embedding + FAISS is the expensive stage.** Encoding ~10M records with
  a small multilingual encoder is very parallelizable on a GPU instance
  (`g5.xlarge` is plenty for a ~118M-param model); batch it, and build a
  separate `IndexFlatIP` (or `IndexIVFFlat` if exact search is too slow at
  this size) per country so you're never searching across country
  boundaries.
- **Feature engineering's row-wise loop in `features.py`** is fine at the
  smoke-test scale but will be a bottleneck at tens of millions of pairs —
  swap the Python loop for `rapidfuzz.process.cdist` in batches, or push it
  into multiprocessing across candidate-pair chunks.
- Consider fine-tuning the embedding model on your own ground-truth pairs
  (contrastive / `MultipleNegativesRankingLoss` with positives from
  `train_ground_truth.tsv`) before the final run — this is usually the
  single highest-leverage change for blocking recall on noisy names, and
  SageMaker Training makes this a cheap experiment once the base pipeline
  works.
- **Stay under the "MIT/Apache, ≤8B params" model constraint** — the
  suggested `intfloat/multilingual-e5-small` (~118M) is MIT-licensed and
  leaves you enormous headroom; LightGBM itself has no parameter-count
  concept the rule would apply to.

## Known gaps / what to tighten next

- `normalize.normalize_address`'s city/state extraction is a heuristic
  (last comma-segment not matching the state/zip) — it's noticeably wrong
  when the city appears *before* the state in the raw string with more
  segments after it (I hit this on real rows in your `train_source2.tsv`,
  e.g. `"GREENSBORO, NC, 19 1/2 STARDUST TRAIL"` extracts `"19 1/2 stardust
  trail"` as the city guess instead of Greensboro). This mainly costs you
  a bit of `city_exact_match` signal — `addr_token_sort_ratio` /
  `addr_partial_ratio` on the full address string are unaffected and carry
  most of the address signal already, but tightening this (e.g. picking
  the *first* non-numeric segment as a city candidate too, and letting the
  feature take the max fuzzy score over a few candidate segments) is a
  cheap win.
- No France-format address handling yet (no gazetteer, no postal-code
  regex) since France isn't in training data to test against — the
  pipeline degrades gracefully (falls back to full-string fuzzy features)
  but won't get the zip/state-exact-match bonus features for French rows.
  Fine, since a French postal-code regex `\b\d{5}\b` is trivial to add to
  `normalize_address` once you want to tune specifically for that slice.
- `pairs_builder.build_labeled_pairs` only trains on pairs blocking
  actually found. Track your blocking recall ceiling separately (fraction
  of ground-truth matches present in `candidate_pairs.tsv`) — that number
  upper-bounds your leaderboard score regardless of classifier quality, and
  is the first thing to report in your methodology doc.
- Consider a light **graph-consistency pass** after thresholding: if S1
  matches both a S2 and a S3 record, and those two records are *also* a
  strong fuzzy match to each other, that's corroborating evidence worth a
  small score boost (and conversely, a lone high-scoring match with no
  such corroboration is a good candidate for a stricter threshold). Not
  implemented here — a natural next iteration once the base pipeline is
  scored on the real leaderboard and you can see where it's losing
  precision vs. recall.

## Methodology doc

`metrics.py`'s `best_threshold_for_f05` curve, `feature_importance.csv`, and
a blocking-recall-ceiling number (compute it once by checking what fraction
of `train_ground_truth.tsv` matches survive into your `candidate_pairs.tsv`)
are the three numbers worth leading the "Candidate generation" and "Model
architecture" sections of `Documentation_template.md` with — they're exactly
what a reviewer needs to sanity-check your pipeline without rerunning it.
