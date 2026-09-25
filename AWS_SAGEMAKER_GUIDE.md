# Running on AWS SageMaker — Complete Guide

## Overview

Your local machine can't handle the full dataset (~1.2 GB raw, ~3-5 GB in memory after normalization). This guide walks you through running the pipeline on AWS SageMaker using the **free-tier** options where possible.

---

## Option A: SageMaker Studio Lab (100% Free — Recommended to Start)

SageMaker Studio Lab is completely free — **no AWS account, no credit card needed.**

| Resource | Limit |
|----------|-------|
| CPU runtime | 12 hrs/session, 4 vCPUs, 16 GB RAM |
| GPU runtime | 4 hrs/session, 1 GPU (T4), 16 GB RAM |
| Storage | 15 GB persistent |

> **16 GB RAM is tight for the full dataset.** You may need to process in chunks or use a sample first. If it doesn't fit, go to Option B.

### Steps

1. **Sign up** at [https://studiolab.sagemaker.aws/](https://studiolab.sagemaker.aws/)
   - Uses a separate account (NOT your AWS account)
   - Approval takes minutes to a few hours

2. **Start a runtime**
   - Click "Start Runtime" → choose **CPU** (for token blocking + LightGBM) or **GPU** (if you need ANN embeddings)
   - Click "Open Project"

3. **Upload the notebook**
   - In JupyterLab, click the upload button (⬆) in the file browser
   - Upload `sagemaker_pipeline.ipynb`

4. **Upload the dataset**
   ```
   # In a terminal tab (File → New → Terminal):
   mkdir -p ~/dataset/train ~/dataset/test
   ```
   - Upload each `.tsv` file to the correct folder via the file browser
   - OR use `wget` if the data is hosted somewhere:
     ```bash
     cd ~/dataset/train
     # wget <your-data-url>/train_source1.tsv
     # ... etc
     ```

5. **Update the data paths** in the notebook's config cell:
   ```python
   DATA_DIR = "/home/studio-lab-user/dataset/train"
   TEST_DIR = "/home/studio-lab-user/dataset/test"
   OUT_DIR  = "/home/studio-lab-user/artifacts"
   ```

6. **Run all cells** (Kernel → Restart Kernel and Run All)

7. **Download results** from the file browser

### Storage tip
The dataset is ~1.5 GB and Studio Lab gives 15 GB. You'll be fine, but don't leave multiple copies around.

---

## Option B: SageMaker Notebook Instance (AWS Free Tier + Pay-as-you-go)

This gives you more control over instance types and storage.

### AWS Free Tier

| Service | Free Tier |
|---------|-----------|
| SageMaker Notebook | 250 hours/month of `ml.t3.medium` (first 2 months) |
| S3 storage | 5 GB |
| Data transfer | 100 GB out |

> ⚠️ `ml.t3.medium` has only **4 GB RAM** — too small for the full dataset. You'll need a larger instance (not free). See cost table below.

### Recommended instances

| Instance | vCPUs | RAM | Cost/hr | Best for |
|----------|-------|-----|---------|----------|
| `ml.t3.medium` | 2 | 4 GB | **Free** (250 hrs) | Testing the notebook with small data |
| `ml.t3.xlarge` | 4 | 16 GB | ~$0.23 | Full token blocking + LightGBM (no ANN) |
| `ml.m5.xlarge` | 4 | 16 GB | ~$0.27 | Same, slightly faster |
| `ml.m5.2xlarge` | 8 | 32 GB | ~$0.54 | Full pipeline with some headroom |
| `ml.g4dn.xlarge` | 4 | 16 GB + GPU | ~$0.74 | ANN embedding (10x faster with GPU) |

**Budget estimate** for a full run:
- Token blocking only (~2-4 hrs): **~$0.50-1.00** on `ml.t3.xlarge`
- With ANN embeddings (~4-8 hrs): **~$3-6** on `ml.g4dn.xlarge`

### Step-by-step setup

#### 1. Create an AWS Account
- Go to [aws.amazon.com](https://aws.amazon.com/)
- Sign up with email + credit card (won't be charged for free tier)
- Enable MFA for security

#### 2. Go to SageMaker
- Sign in to AWS Console
- Search for "SageMaker" → click "Amazon SageMaker"
- Choose your region (e.g., `us-east-1` has the most free-tier capacity)

#### 3. Create a Notebook Instance
1. In the left sidebar: **Notebook** → **Notebook instances**
2. Click **"Create notebook instance"**
3. Fill in:
   - **Name**: `entity-resolution`
   - **Instance type**: `ml.t3.medium` (free tier) or `ml.t3.xlarge` (16 GB, recommended)
   - **Volume size**: `30 GB` (default is 5 — increase it for the dataset)
   - **IAM role**: Create a new role → allow S3 access
   - Leave everything else default
4. Click **"Create notebook instance"**
5. Wait 2-3 minutes for status to become **"InService"**

#### 4. Upload the Dataset

**Option 1: Direct upload via JupyterLab**
1. Click "Open JupyterLab" on your notebook instance
2. Open a terminal: File → New → Terminal
3. Create directories:
   ```bash
   mkdir -p ~/SageMaker/dataset/train ~/SageMaker/dataset/test
   ```
4. Use the JupyterLab file browser to upload `.tsv` files to the correct folders

**Option 2: Upload via S3 (better for large files)**
1. Go to S3 in AWS Console
2. Create a bucket: `my-ml-challenge-data`
3. Upload your `.tsv` files to:
   ```
   s3://my-ml-challenge-data/train/train_source1.tsv
   s3://my-ml-challenge-data/train/train_source2.tsv
   s3://my-ml-challenge-data/train/train_source3.tsv
   s3://my-ml-challenge-data/train/train_ground_truth.tsv
   s3://my-ml-challenge-data/test/test_source1.tsv
   s3://my-ml-challenge-data/test/test_source2.tsv
   s3://my-ml-challenge-data/test/test_source3.tsv
   ```
4. In the notebook terminal, download from S3:
   ```bash
   mkdir -p ~/SageMaker/dataset/train ~/SageMaker/dataset/test
   aws s3 cp s3://my-ml-challenge-data/train/ ~/SageMaker/dataset/train/ --recursive
   aws s3 cp s3://my-ml-challenge-data/test/ ~/SageMaker/dataset/test/ --recursive
   ```

#### 5. Upload & Run the Notebook
1. Upload `sagemaker_pipeline.ipynb` via the JupyterLab file browser
2. Open the notebook
3. **Verify the data paths** in the config cell:
   ```python
   DATA_DIR = "/home/ec2-user/SageMaker/dataset/train"
   TEST_DIR = "/home/ec2-user/SageMaker/dataset/test"
   OUT_DIR  = "/home/ec2-user/SageMaker/artifacts"
   ```
4. Select kernel: `conda_python3`
5. Run all cells: **Kernel → Restart Kernel and Run All**

#### 6. Download Results
- Use the file browser to navigate to `artifacts/output/`
- Right-click → Download:
  - `candidate_pairs.tsv`
  - `matching_results.tsv`
- Or download the zip: `artifacts/submission.zip`

#### 7. STOP the Instance (Important!)
- Go back to SageMaker Console → Notebook instances
- Select your instance → **Actions → Stop**
- **You are billed per hour while the instance is running!**

---

## Option C: SageMaker Studio (Newer UI)

SageMaker Studio is AWS's newer IDE. It's more complex to set up but offers JupyterLab 3 and more features.

### Quick setup
1. SageMaker Console → **Studio** → **Create domain** (use Quick Setup)
2. Launch Studio → Upload notebook
3. Choose a kernel image with Python 3.10+ and appropriate instance
4. Data paths: `/home/sagemaker-user/dataset/...`

Use this only if you're already familiar with SageMaker Studio.

---

## Running the Pipeline — What to Expect

### Phase 1: Token Blocking Only (Recommended First Run)

Set `USE_ANN = False` in the config cell. This skips the slow embedding step.

```python
USE_ANN = False  # Skip ANN for fast first run
```

| Step | Time (est.) | Memory |
|------|-------------|--------|
| Load + normalize | 5-15 min | ~4 GB |
| Token blocking | 5-20 min | ~3 GB peak |
| Feature engineering | 10-30 min | ~2 GB |
| LightGBM training | 2-5 min | ~1 GB |
| Threshold tuning | 1 min | ~1 GB |

**Total**: ~30-70 minutes on `ml.t3.xlarge`

### Phase 2: With ANN Embeddings

Set `USE_ANN = True`. Use a GPU instance (`ml.g4dn.xlarge`) for faster encoding.

| Step | Time (est.) | Notes |
|------|-------------|-------|
| ANN S1 x S2 | 15-45 min | Encoding ~7M texts |
| ANN S1 x S3 | 15-45 min | Encoding ~7M texts |

**Total**: ~60-150 minutes additional

### Memory Tips

If you run out of memory:

1. **Process per-country** (already built in — the code loops over countries)
2. **Reduce candidate caps**:
   ```python
   CANDIDATE_CAP_LEXICAL = 20  # instead of 40
   CANDIDATE_CAP_ANN = 10      # instead of 20
   ```
3. **Increase blocking strictness**:
   ```python
   MAX_TOKEN_BLOCK_PAIRS = 1000  # instead of 5000
   ```
4. **Use a larger instance** (`ml.m5.2xlarge` = 32 GB)
5. **Delete intermediate DataFrames** after each step:
   ```python
   del token_cand2, token_cand3  # after capping is done
   import gc; gc.collect()
   ```

---

## Experiment Ablation Guide

The notebook has experiment flags at the top. Here's how to run ablations:

### Experiment 1: Baseline (token blocking only)
```python
USE_TOKEN_BLOCKING = True
USE_ANN = False
USE_CHAR_NGRAM_FEATURES = False
```

### Experiment 2: + Char N-gram Features
```python
USE_CHAR_NGRAM_FEATURES = True  # only change from Exp 1
```

### Experiment 3: + ANN
```python
USE_ANN = True  # needs GPU instance for speed
```

### Experiment 4: + Separate Name ANN
```python
USE_SEPARATE_NAME_ANN = True
```

### Experiment 5: Different Blocking Threshold
```python
MAX_TOKEN_BLOCK_PAIRS = 10000  # try 1000, 5000, 10000
```

**Compare** the `val macro F0.5` from each run and the `candidate recall` numbers.

---

## Troubleshooting

### "KernelDied" or "MemoryError"
- Your instance is too small. Switch to `ml.m5.2xlarge` (32 GB)
- Or reduce `CANDIDATE_CAP_LEXICAL` and `MAX_TOKEN_BLOCK_PAIRS`

### "ModuleNotFoundError: No module named 'xxx'"
- Re-run the first code cell (pip install)
- Make sure you're using the right kernel

### "FileNotFoundError"
- Check your `DATA_DIR` path. Open a terminal and run `ls ~/SageMaker/dataset/train/`

### ANN is very slow
- Use a GPU instance (`ml.g4dn.xlarge`)
- Or reduce `ANN_BATCH` if you get GPU OOM

### How to resume after a crash
- The notebook saves artifacts (`model.txt`, `threshold.json`) at each major step
- For inference, you can re-load the model without retraining:
  ```python
  booster = lgb.Booster(model_file=os.path.join(OUT_DIR, "model.txt"))
  ```

---

## Cost Summary

| Scenario | Instance | Time | Cost |
|----------|----------|------|------|
| Test with small data | `ml.t3.medium` | 1 hr | **Free** |
| Token blocking only | `ml.t3.xlarge` | 2 hrs | ~$0.50 |
| Full pipeline (no GPU) | `ml.m5.2xlarge` | 4 hrs | ~$2.00 |
| Full pipeline (GPU ANN) | `ml.g4dn.xlarge` | 6 hrs | ~$4.50 |

> **Always stop your instance when done!** A running `ml.g4dn.xlarge` costs ~$18/day.

---

## Files in This Project

| File | Purpose |
|------|---------|
| `sagemaker_pipeline.ipynb` | **Upload this to SageMaker** — complete self-contained pipeline |
| `generate_notebook.py` | Script that generated the notebook (for reference/modification) |
| `config.py` | Local config (not needed on SageMaker — everything is in the notebook) |
| `blocking.py` | Local blocking module (inlined in notebook) |
| `normalize.py` | Local normalization module (inlined in notebook) |
| `features.py` | Local features module (inlined in notebook) |
| `train.py` | Local training script (replaced by notebook) |
| `infer.py` | Local inference script (replaced by notebook) |
