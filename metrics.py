"""F_0.5 macro-average scorer, exactly as defined in the problem statement:
computed per Source-1 entity (singletons included, predicting empty
correctly = 1.0), then averaged across all Source-1 entities.

Also includes candidate-recall evaluation for diagnosing whether candidate
generation is throwing away true matches before LightGBM can score them.
"""
from collections import defaultdict


def _parse_ids(s):
    if s is None or (isinstance(s, float)):
        return set()
    s = str(s).strip()
    if not s:
        return set()
    return set(s.split(","))


def macro_f05(predictions: dict, ground_truth: dict, all_source1_ids) -> float:
    """
    predictions: {source1_entity_id: set_or_comma_str of predicted other ids}
    ground_truth: {source1_entity_id: set_or_comma_str of true other ids}
    all_source1_ids: iterable of every S1 id that must be scored (missing
        entities in `predictions` are treated as an empty prediction)
    """
    scores = []
    for sid in all_source1_ids:
        pred = predictions.get(sid, set())
        true = ground_truth.get(sid, set())
        if not isinstance(pred, set):
            pred = _parse_ids(pred)
        if not isinstance(true, set):
            true = _parse_ids(true)

        if not true and not pred:
            scores.append(1.0)
            continue
        if not pred:
            scores.append(0.0)  # recall=0 -> F=0 whenever there's a true match to find
            continue
        if not true:
            scores.append(0.0)  # any prediction on a true singleton is a false merge
            continue

        tp = len(pred & true)
        precision = tp / len(pred)
        recall = tp / len(true)
        if precision == 0 and recall == 0:
            scores.append(0.0)
            continue
        f05 = (1.25 * precision * recall) / (0.25 * precision + recall) if (0.25 * precision + recall) > 0 else 0.0
        scores.append(f05)

    return sum(scores) / len(scores) if scores else 0.0


def best_threshold_for_f05(scored_pairs, ground_truth: dict, all_source1_ids,
                            thresholds=None):
    """Sweep thresholds on model probability to maximize macro F0.5.

    scored_pairs: DataFrame with columns source1_entity_id, other_entity_id, score
    Returns (best_threshold, best_f05, curve) where curve is a list of (t, f05).

    Also reports precision, recall, F1, and predicted match count at the best threshold.
    """
    import numpy as np
    if thresholds is None:
        thresholds = np.arange(0.05, 0.96, 0.02)

    curve = []
    best_t, best_f = None, -1.0
    for t in thresholds:
        kept = scored_pairs[scored_pairs["score"] >= t]
        preds = defaultdict(set)
        for sid, oid in zip(kept["source1_entity_id"], kept["other_entity_id"]):
            preds[sid].add(oid)
        f = macro_f05(preds, ground_truth, all_source1_ids)
        curve.append((float(t), f))
        if f > best_f:
            best_f, best_t = f, float(t)

    # --- Extended reporting at best threshold ---
    if best_t is not None:
        kept = scored_pairs[scored_pairs["score"] >= best_t]
        preds = defaultdict(set)
        for sid, oid in zip(kept["source1_entity_id"], kept["other_entity_id"]):
            preds[sid].add(oid)

        total_tp, total_pred, total_true = 0, 0, 0
        for sid in all_source1_ids:
            pred = preds.get(sid, set())
            true = ground_truth.get(sid, set())
            if not isinstance(true, set):
                true = _parse_ids(true)
            total_tp += len(pred & true)
            total_pred += len(pred)
            total_true += len(true)

        precision = total_tp / total_pred if total_pred > 0 else 0.0
        recall = total_tp / total_true if total_true > 0 else 0.0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        print(f"\n  === Threshold Tuning Results ===")
        print(f"  Best threshold: {best_t:.4f}")
        print(f"  Macro F0.5:     {best_f:.4f}")
        print(f"  Micro Precision: {precision:.4f}")
        print(f"  Micro Recall:    {recall:.4f}")
        print(f"  Micro F1:        {f1:.4f}")
        print(f"  Predicted matches: {total_pred}")
        print(f"  True matches:      {total_true}")
        print(f"  True positives:    {total_tp}")

    return best_t, best_f, curve


def candidate_recall(candidate_pairs, ground_truth: dict, source1_ids=None) -> dict:
    """Measure what fraction of ground-truth matches are present in the
    candidate set (before LightGBM scoring).

    This is the recall ceiling — no classifier can recover matches that
    were discarded during candidate generation.

    candidate_pairs: DataFrame with source1_entity_id, other_entity_id
    ground_truth: {source1_entity_id: set of matched entity ids}
    source1_ids: optional subset of S1 ids to evaluate (e.g. val split)

    Returns dict with recall stats.
    """
    # Build candidate lookup
    cand_set = defaultdict(set)
    for sid, oid in zip(candidate_pairs["source1_entity_id"],
                         candidate_pairs["other_entity_id"]):
        cand_set[sid].add(oid)

    if source1_ids is None:
        source1_ids = set(ground_truth.keys())

    total_true_matches = 0
    found_matches = 0
    entities_with_matches = 0
    entities_fully_recalled = 0
    entities_with_missed = 0

    for sid in source1_ids:
        true_ids = ground_truth.get(sid, set())
        if not isinstance(true_ids, set):
            true_ids = _parse_ids(true_ids)
        if not true_ids:
            continue

        entities_with_matches += 1
        candidates = cand_set.get(sid, set())
        found = len(true_ids & candidates)
        total_true_matches += len(true_ids)
        found_matches += found

        if found == len(true_ids):
            entities_fully_recalled += 1
        elif found < len(true_ids):
            entities_with_missed += 1

    recall = found_matches / total_true_matches if total_true_matches > 0 else 0.0

    return {
        "candidate_recall": recall,
        "found_matches": found_matches,
        "total_true_matches": total_true_matches,
        "entities_with_matches": entities_with_matches,
        "entities_fully_recalled": entities_fully_recalled,
        "entities_with_missed": entities_with_missed,
        "total_candidates": len(candidate_pairs),
    }


def evaluate_candidate_recall_at_k(candidate_pairs, score_col, ground_truth: dict,
                                     k_values=(20, 50, 100, 200, 500),
                                     source1_ids=None):
    """Evaluate candidate recall at different cap values K.

    For each K, caps the candidates per entity to the top-K by score_col,
    then measures recall.

    Prints a formatted report and returns a list of (K, stats) tuples.
    """
    import time

    if source1_ids is not None:
        candidate_pairs = candidate_pairs[
            candidate_pairs["source1_entity_id"].isin(source1_ids)
        ]

    results = []
    print("\n  === Candidate Recall at Various K ===")
    print(f"  {'K':>6}  {'Candidates':>12}  {'Recall':>8}  {'Found':>8}  {'Total':>8}  {'Time':>8}")
    print(f"  {'─'*6}  {'─'*12}  {'─'*8}  {'─'*8}  {'─'*8}  {'─'*8}")

    # Full (uncapped) recall first
    t0 = time.time()
    stats_full = candidate_recall(candidate_pairs, ground_truth, source1_ids)
    dt = time.time() - t0
    print(f"  {'ALL':>6}  {stats_full['total_candidates']:>12,}  "
          f"{stats_full['candidate_recall']:>8.4f}  "
          f"{stats_full['found_matches']:>8,}  "
          f"{stats_full['total_true_matches']:>8,}  "
          f"{dt:>7.1f}s")
    results.append(("ALL", stats_full))

    for k in sorted(k_values):
        t0 = time.time()
        if score_col in candidate_pairs.columns:
            capped = (
                candidate_pairs.sort_values(score_col, ascending=False)
                .groupby("source1_entity_id", group_keys=False)
                .head(k)
            )
        else:
            capped = candidate_pairs.groupby(
                "source1_entity_id", group_keys=False
            ).head(k)

        stats = candidate_recall(capped, ground_truth, source1_ids)
        dt = time.time() - t0
        print(f"  {k:>6}  {stats['total_candidates']:>12,}  "
              f"{stats['candidate_recall']:>8.4f}  "
              f"{stats['found_matches']:>8,}  "
              f"{stats['total_true_matches']:>8,}  "
              f"{dt:>7.1f}s")
        results.append((k, stats))

    return results
