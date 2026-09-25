"""F_0.5 macro-average scorer, exactly as defined in the problem statement:
computed per Source-1 entity (singletons included, predicting empty
correctly = 1.0), then averaged across all Source-1 entities.
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
    return best_t, best_f, curve
