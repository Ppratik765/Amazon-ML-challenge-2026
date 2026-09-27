import pandas as pd
import numpy as np


def macro_f_0_5(gt_dict, pred_dict):
    """Reference implementation -- unchanged from the original. Deliberately kept
    simple (a plain Python loop) so it stays obviously correct; it's the function
    used to report the final, trusted number. build_grid_search_table +
    score_from_masks below are a vectorized re-implementation of the same rules,
    used only to make threshold grid search fast -- their agreement with this
    function is checked directly in the smoke test rather than assumed."""
    scores = []
    for s1, gt_matches in gt_dict.items():
        pred_matches = pred_dict.get(s1, set())
        
        if len(gt_matches) == 0:
            if len(pred_matches) == 0:
                scores.append(1.0)
            else:
                scores.append(0.0)
        else:
            tp = len(gt_matches.intersection(pred_matches))
            fp = len(pred_matches - gt_matches)
            fn = len(gt_matches - pred_matches)
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            
            if precision == 0 and recall == 0:
                f_0_5 = 0.0
            else:
                f_0_5 = (1.25 * precision * recall) / (0.25 * precision + recall)
            scores.append(f_0_5)
            
    return sum(scores) / len(scores) if scores else 0.0


def parse_gt_file(gt_df):
    """train_ground_truth.tsv -> {source1_entity_id: set(matched_entity_ids)},
    including entities with an empty matched_entity_ids field (empty set)."""
    gt_dict = {}
    for _, row in gt_df.iterrows():
        s1 = row['source1_entity_id']
        if pd.notna(row.get('matched_entity_ids')) and str(row['matched_entity_ids']).strip():
            gt_dict[s1] = set(str(row['matched_entity_ids']).split(','))
        else:
            gt_dict[s1] = set()
    return gt_dict


def build_scoring_table(pairs_df, gt_dict, all_s1_ids):
    """
    Builds the union, per S1 entity, of every candidate pair produced by blocking
    AND every true match from gt_dict -- including true matches blocking never
    retrieved at all (those get prob_rank_in_source = +inf so no threshold/cap
    combination can ever predict them; they count as a guaranteed false negative
    for every combo, exactly as they would in the real metric). This table is
    built once; grid search below only recomputes the boolean "is_pred" mask per
    threshold combo against columns that don't change, which is what makes the
    grid fast.

    Also includes every S1 entity with literally zero candidate pairs and zero gt
    matches (contributes a trivial score of 1.0), so the macro average matches
    what the real evaluator would compute over the full required S1 id set.
    """
    base = pairs_df[['source1_entity_id', 'candidate_entity_id', 'target_source', 'prob']].copy()
    base['is_gt'] = [
        cid in gt_dict.get(s1, set())
        for s1, cid in zip(base['source1_entity_id'], base['candidate_entity_id'])
    ]

    covered = set(zip(base['source1_entity_id'], base['candidate_entity_id']))
    missing_rows = []
    for s1, matches in gt_dict.items():
        for cid in matches:
            if (s1, cid) not in covered:
                missing_rows.append({
                    'source1_entity_id': s1,
                    'candidate_entity_id': cid,
                    'target_source': 1 if str(cid).startswith('S2') else 2,
                    'prob': -1.0,
                    'is_gt': True,
                })
    if missing_rows:
        base = pd.concat([base, pd.DataFrame(missing_rows)], ignore_index=True)

    base['prob_rank_in_source'] = base.groupby(['source1_entity_id', 'target_source'])['prob'] \
        .rank(method='first', ascending=False)

    zero_candidate_ids = set(all_s1_ids) - set(base['source1_entity_id'].unique())
    return base, zero_candidate_ids


def score_thresholds(base, zero_candidate_ids, gt_dict, threshold_s2, threshold_s3, cap=1):
    """Vectorized macro F0.5 for one (threshold_s2, threshold_s3, cap) combo,
    following exactly the same per-entity rules as macro_f_0_5 above."""
    thresh = np.where(base['target_source'].values == 1, threshold_s2, threshold_s3)
    is_pred = (base['prob_rank_in_source'].values <= cap) & (base['prob'].values >= thresh)
    is_gt = base['is_gt'].values

    tp = is_pred & is_gt
    fp = is_pred & ~is_gt
    fn = ~is_pred & is_gt

    g = pd.DataFrame({'s1': base['source1_entity_id'].values, 'tp': tp, 'fp': fp, 'fn': fn}) \
        .groupby('s1').sum()

    precision = g['tp'] / (g['tp'] + g['fp']).replace(0, np.nan)
    recall = g['tp'] / (g['tp'] + g['fn']).replace(0, np.nan)
    denom = (0.25 * precision.fillna(0) + recall.fillna(0))
    f_0_5 = np.where(denom > 0, (1.25 * precision.fillna(0) * recall.fillna(0)) / denom.replace(0, np.nan), 0.0)
    f_0_5 = pd.Series(f_0_5, index=g.index).fillna(0.0)

    has_gt = np.array([len(gt_dict.get(s1, set())) > 0 for s1 in g.index])
    no_gt_no_pred = (~has_gt) & ((g['tp'] + g['fp']) == 0)
    f_0_5 = f_0_5.where(has_gt, other=np.where(no_gt_no_pred, 1.0, 0.0))

    n_zero_cand_trivial = sum(1 for s1 in zero_candidate_ids if len(gt_dict.get(s1, set())) == 0)
    n_zero_cand_missed = len(zero_candidate_ids) - n_zero_cand_trivial

    total_score = f_0_5.sum() + n_zero_cand_trivial * 1.0 + n_zero_cand_missed * 0.0
    total_n = len(f_0_5) + len(zero_candidate_ids)
    return total_score / total_n if total_n else 0.0


def resolve_conflicts(scored_df, threshold_s2, threshold_s3, cap=1):
    """
    Source 1 is explicitly the deduplicated reference set -- every real-world
    business appears exactly once there. That means a given Source-2 or
    Source-3 record can genuinely belong to at most one Source-1 entity, even
    though one Source-1 entity can rightly claim several Source-2/Source-3
    records (duplicates *within* S2/S3 of the same real business). Independent
    per-S1 thresholding (score_thresholds above) can still let two different S1
    entities both claim the same candidate above threshold -- F0.5 then scores
    that as a false positive on whichever one is wrong.

    This computes the same independent top-cap-within-its-own-group selection
    score_thresholds/grid_search_thresholds optimize, then ONLY REMOVES entries
    from it to break ties on a shared candidate (highest confidence keeps the
    claim) -- it never substitutes in a different pick for the loser (an earlier
    version did, by sorting globally and letting a loser fall through to a
    lower-ranked-in-its-own-group candidate; that let it change WHICH pairs are
    considered, not just resolve a tie, and could turn a correct empty/singleton
    prediction into a wrong one).

    IMPORTANT -- this is a bet, not a guaranteed improvement: it trusts the
    model's own confidence ordering to decide who keeps a disputed candidate.
    When that ordering is right, this only removes duplicate false positives and
    strictly helps. When two S1 entities compete for a candidate that is truly
    only one of theirs, and the model happens to be MORE confident about the
    WRONG claim than the RIGHT one (a real calibration error, not a hypothetical
    one -- verified directly: it happens on both a real leave-one-country-out
    diagnostic run and in randomized testing here), this can steal a genuine
    true positive from its rightful entity and give the macro-averaged F0.5 a
    net loss on that swap, even though it only ever removes entries. Do not
    apply this by default -- compare macro_f_0_5 with and without it on your own
    holdout split (train.py does this, and picks whichever wins; see
    USE_CONFLICT_RESOLUTION in thresholds.json) rather than assuming it helps.

    Returns {source1_entity_id: set(candidate_entity_id)} -- NOT filled in with
    empty sets for every required S1 id; callers should reindex against the
    full required id list themselves (macro_f_0_5's gt_dict.items() loop and
    pipeline.py's final reindex both already do this).
    """
    df = scored_df.copy()
    df['thresh'] = np.where(df['target_source'].values == 1, threshold_s2, threshold_s3)
    df = df[df['prob'].values >= df['thresh'].values]
    if df.empty:
        return {}

    # The independent selection: same rule score_thresholds uses. Restricting to
    # this set BEFORE resolving conflicts is what makes this a pure removal.
    df['rank_in_source'] = df.groupby(['source1_entity_id', 'target_source'])['prob'] \
        .rank(method='first', ascending=False)
    df = df[df['rank_in_source'].values <= cap]
    if df.empty:
        return {}

    df = df.sort_values('prob', ascending=False)
    claimed_candidates = set()
    pred_dict = {}
    for s1, cid in zip(df['source1_entity_id'].values, df['candidate_entity_id'].values):
        if cid in claimed_candidates:
            continue
        claimed_candidates.add(cid)
        pred_dict.setdefault(s1, set()).add(cid)
    return pred_dict


def grid_search_thresholds(pairs_df, gt_dict, all_s1_ids,
                            threshold_grid=None, caps=(1, 2, 3, 5, 10)):
    """Coarse-to-reasonable grid search over (threshold_s2, threshold_s3, cap)
    maximizing macro F0.5 on a validation split. The problem statement says an S1
    entity may match "zero, one, or many" records per source, so cap is a real
    hyperparameter to tune, not fixed at 1 -- a high cap combined with a strict
    probability threshold can behave close to an uncapped flat-threshold rule,
    while a low cap plus a looser threshold behaves close to a top-k rule; the grid
    lets the data decide which region wins on your validation split. Returns
    (best_threshold_s2, best_threshold_s3, best_cap, best_score, all_results)."""
    if threshold_grid is None:
        threshold_grid = np.round(np.arange(0.05, 0.96, 0.025), 4)

    base, zero_candidate_ids = build_scoring_table(pairs_df, gt_dict, all_s1_ids)

    results = []
    best = (0.5, 0.5, 1, -1.0)
    for cap in caps:
        for t2 in threshold_grid:
            for t3 in threshold_grid:
                score = score_thresholds(base, zero_candidate_ids, gt_dict, t2, t3, cap)
                results.append((t2, t3, cap, score))
                if score > best[3]:
                    best = (t2, t3, cap, score)
    return best[0], best[1], best[2], best[3], results
