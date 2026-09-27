# Fast TF-IDF Pipeline

This pipeline uses sparse TF-IDF cosine similarity for candidate generation instead of neural embeddings.

## Why this is better/faster:
Neural embeddings (like SentenceTransformers + FAISS) take a very long time to process millions of records and use a lot of memory. 

By using character-level n-grams with TF-IDF, we can capture spelling mistakes, reorderings, and abbreviations extremely well and calculate cosine similarity almost instantly using a highly optimized sparse matrix dot product. We chunk the computation to ensure it comfortably fits into RAM.

As requested, we sub-sample Source 1 during training (cap at 400,000 rows), which provides plenty of positive and negative pairs for LightGBM to learn optimal thresholds and splits without needing to train on 2.2 million rows. Inference is naturally done on the full test set.

## Running the Pipeline

From the `tfidf_pipeline` directory, or by specifying paths, you can run:

```bash
python tfidf_pipeline/train_fast.py --data-dir student_resource/dataset --out-dir output_tfidf
```

### Steps Executed Automatically:
1. **Load and Normalize**: Normalizes 400k training records in parallel.
2. **TF-IDF Blocking**: Builds TF-IDF matrices and runs sparse top-K dot product.
3. **Feature Engineering**: Runs the same fast token matching logic.
4. **LightGBM**: Trains the F_0.5 optimized model.
5. **Inference**: Applies the pipeline to the full test set and generates `candidate_pairs.tsv` and `matching_results.tsv` in `output_tfidf/submission/`.

You can take this output folder and zip it directly!
