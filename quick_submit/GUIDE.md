# Quick Submit — How to Run

> **Updated after v2 overhaul** — blocking now includes Soundex phonetic + character-trigram
> strategies alongside token/zip/city. Two new LightGBM features added. Caps raised.

---

## ⏰ I Have Limited Time — What Do I Run?

| Time Left | Command | Expected Score |
|---|---|---|
| < 2 hrs | `--infer-only` with existing model + relaxed caps | Marginal gain on existing run |
| 2–4 hrs | Default (token + phonetic + trigram), no ANN | **~0.55–0.62** |
| 4–7 hrs | `--embeddings` (+ combined ANN) | **~0.62–0.70** |
| 7–14 hrs | `--full` (+ combined + name-only ANN) | **~0.67–0.75** |

> **Recommendation for EOD deadline**: Start `--full --fresh` now. It trains on new phonetic+trigram
> blocking candidates with 2 new features, and runs ANN on top. This is your best shot.

---

## TL;DR — Run This Right Now

```bash
# From the Amazon ML Challenge root:
.\.venv\Scripts\python.exe quick_submit/quick_run.py \
    --data-dir student_resource/dataset \
    --out-dir ./output \
    --full \
    --fresh
```

**Or on SageMaker:**
```bash
python quick_submit/quick_run.py \
    --data-dir dataset \
    --out-dir ./output \
    --full \
    --fresh
```

---

## What Changed in v2 (Today's Update)

### Blocking — 2 new strategies (higher recall ceiling)

| Strategy | What it catches |
|---|---|
| Token blocking (existing) | Shared name tokens within same country |
| Zip blocking (existing) | Same ZIP/PIN code |
| City + first-token (existing) | Same city + first word |
| **Soundex phonetic** ✨ NEW | McDonald↔MacDonald, Sharma↔Sarma, Singh↔Sing |
| **Char-trigram Jaccard** ✨ NEW | Heavy typos: `caloce`↔`calosa`, `walmartt`↔`walmart` |

### Features — 2 new (22 total)

| Feature | Description |
|---|---|
| `name_trigram_jaccard` ✨ | Jaccard similarity of char 3-grams of normalized names |
| `name_soundex_match` ✨ | 1.0 if both names have the same Soundex code |

### Settings raised

| Setting | Before | After |
|---|---|---|
| `S1_TRAIN_CAP` | 80k | 120k |
| `CAP_LEXICAL` | 30 | 50 |
| `CAP_ANN` | 15 | 25 |
| `MAX_BLOCK_PAIRS` | 5k | 8k |
| `num_leaves` (LightGBM) | 63 | 127 |
| `num_boost_round` | 1500 | 2000 |

> **Important:** Because features changed, you **MUST** use `--fresh` to clear old checkpoints.
> The old `model.txt` is incompatible with the new 22-feature schema.

---

## Three Speed Modes

### Mode 1 — Token + Phonetic + Trigram (DEFAULT, no ANN)
**~2–3 hr train + ~2–3 hr infer = ~4–6 hr total**

Best for: "I have 5-6 hours and no GPU."

```bash
.\.venv\Scripts\python.exe quick_submit/quick_run.py \
    --data-dir student_resource/dataset \
    --out-dir ./output \
    --fresh
```

---

### Mode 2 — + Combined ANN embeddings (`--embeddings`)
**~4–5 hr train + ~4–6 hr infer = ~8–11 hr total**

Best for: Overnight / ~10 hr window. Adds multilingual-E5 ANN on top of all lexical blocking.
Catches Devanagari↔Latin matches and heavy-typo cases that even trigram blocking misses.

```bash
.\.venv\Scripts\python.exe quick_submit/quick_run.py \
    --data-dir student_resource/dataset \
    --out-dir ./output \
    --embeddings \
    --fresh
```

---

### Mode 3 — Full pipeline (`--full`) ⭐ RECOMMENDED
**~5–7 hr train + ~7–10 hr infer = ~12–17 hr total**

Best for: Maximum score. Adds both combined AND name-only ANN on top of all lexical blocking.

```bash
.\.venv\Scripts\python.exe quick_submit/quick_run.py \
    --data-dir student_resource/dataset \
    --out-dir ./output \
    --full \
    --fresh
```

---

## Resuming After a Crash

Just **run the exact same command without `--fresh`**. Completed stages are automatically skipped.

```bash
# Resume (no --fresh)
.\.venv\Scripts\python.exe quick_submit/quick_run.py \
    --data-dir student_resource/dataset \
    --out-dir ./output \
    --full
```

---

## Inference Only (Re-run Inference with Existing Model)

If the model is already trained and you just want to re-run inference (e.g., with relaxed caps):

```bash
.\.venv\Scripts\python.exe quick_submit/quick_run.py \
    --data-dir student_resource/dataset \
    --out-dir ./output \
    --full \
    --infer-only \
    --fresh
```

> `--fresh` here only clears inference checkpoints (not the model). Use this when you want
> to redo blocking/features on test data with updated settings but keep the trained model.

---

## Validating Your Submission Before Upload

Use the provided validator (no dependencies):

```bash
cd student_resource
python utils/validate_submission.py \
    --matching ../output/submission/matching_results.tsv \
    --candidate ../output/submission/candidate_pairs.tsv \
    --test-dir dataset/test
```

`PASS` = safe to upload. Fix any listed issues before submitting.

---

## Expected Output Structure

```
output/
├── model.txt               ← Trained LightGBM model (22 features)
├── threshold.json          ← Tuned decision threshold + val F0.5
├── checkpoints/
│   ├── train/              ← Train stage checkpoints (for resume)
│   └── infer/              ← Inference stage checkpoints (for resume)
└── submission/
    ├── candidate_pairs.tsv     ← SUBMIT THIS (blocking candidates)
    └── matching_results.tsv    ← SUBMIT THIS (final predictions)
```

---

## Data Directory Structure

```
student_resource/dataset/
├── train/
│   ├── train_source1.tsv
│   ├── train_source2.tsv
│   ├── train_source3.tsv
│   └── train_ground_truth.tsv
└── test/
    ├── test_source1.tsv
    ├── test_source2.tsv
    └── test_source3.tsv
```

Pass the **parent** `dataset/` directory as `--data-dir`. The script auto-discovers `train/` and `test/` subdirs.

---

## Why the Previous Run Scored 0.47

The token-only blocker has a **hard recall ceiling** — it silently misses:

1. **Phonetically similar names** (McDonald vs MacDonald) — now caught by Soundex blocking
2. **Heavy-typo names with no shared full token** (caloce vs calosa) — now caught by trigram blocking  
3. **Cross-script names** (Devanagari S2 vs Latin S1) — still only caught by ANN (`--embeddings`/`--full`)
4. **France test entities** — the test set has French businesses; the token blocker handles them
   the same as US/India (language-agnostic), but ANN embedding (multilingual-E5) handles them best

The LightGBM model can only predict matches from candidates the blocker surfaced.
If the blocker misses a true match, the score for that entity is 0 regardless of model quality.

---

## Tips for SageMaker

```bash
# Always run in tmux to survive browser disconnects
tmux new -s run
# paste your command
# Ctrl+B then D to detach
# tmux attach -t run to reattach

# Monitor memory
watch -n5 free -h

# Check disk space (checkpoints can be 5-10 GB)
df -h .
```

The script uses all available CPU cores automatically (`NUM_WORKERS = min(cpu_count, 16)`).
