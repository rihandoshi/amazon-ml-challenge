"""Construct a labeled pair dataset for the pairwise classifier.

Positives: every (S1, matched-id) pair straight from ground truth.
Negatives: everything else that blocking surfaced as a candidate for that
S1 entity but that ground truth does NOT list as a match. These are
"hard" negatives by construction — they passed blocking, i.e. they share a
name token / zip / city+first-token with the true entity, which is exactly
the kind of near-miss the classifier needs to learn to reject for the
precision-heavy F0.5 metric.

If a true match was missed by blocking entirely (a recall gap), it's
dropped from training here — you should separately track and report your
blocking recall ceiling (see how_to_run.md) since it caps your achievable
score regardless of the classifier.
"""
import pandas as pd


def build_labeled_pairs(candidate_pairs: pd.DataFrame, ground_truth: pd.DataFrame,
                         max_negatives_per_entity: int = 20, random_state: int = 42) -> pd.DataFrame:
    """
    candidate_pairs: columns source1_entity_id, other_entity_id (blocking output)
    ground_truth: columns source1_entity_id, matched_entity_ids (comma string)

    Returns candidate_pairs with an added `label` column (0/1), negatives
    subsampled per entity to keep the dataset from being dominated by the
    (usually huge) majority of true negatives.
    """
    gt_map = {}
    for sid, ids in zip(ground_truth["source1_entity_id"], ground_truth["matched_entity_ids"]):
        ids = str(ids).strip()
        gt_map[sid] = set(ids.split(",")) if ids else set()

    def is_match(row):
        return 1 if row["other_entity_id"] in gt_map.get(row["source1_entity_id"], set()) else 0

    cp = candidate_pairs.copy()
    cp["label"] = cp.apply(is_match, axis=1)

    positives = cp[cp["label"] == 1]
    negatives = cp[cp["label"] == 0]

    # cap negatives per S1 entity so entities with huge candidate sets don't dominate.
    # (Deliberately not groupby(...).apply(sample) — recent pandas versions drop the
    # grouping column from the apply result, which silently corrupts source1_entity_id.)
    neg_parts = []
    for _, g in negatives.groupby("source1_entity_id", sort=False):
        n = min(len(g), max_negatives_per_entity)
        neg_parts.append(g.sample(n=n, random_state=random_state))
    negatives_capped = pd.concat(neg_parts, ignore_index=True) if neg_parts else negatives

    out = pd.concat([positives, negatives_capped], ignore_index=True)
    return out.sample(frac=1.0, random_state=random_state).reset_index(drop=True)


def group_split_entities(entity_ids, val_frac: float = 0.15, random_state: int = 42):
    """Split unique S1 entity ids into train/val sets (group split — every
    pair for a given S1 goes entirely to one side, so the model can't
    leak entity-specific info across the split).
    """
    import numpy as np
    rng = np.random.default_rng(random_state)
    ids = np.array(sorted(set(entity_ids)))
    rng.shuffle(ids)
    n_val = int(len(ids) * val_frac)
    val_ids = set(ids[:n_val])
    train_ids = set(ids[n_val:])
    return train_ids, val_ids
