import os
import json
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from src.preprocess import preprocess_df
from src.blocking import BlockingEngine, cap_candidates_per_source, enrich_pair_scores
from src.features import FeatureExtractor
from src.model import EnsembleModel
from src.evaluate import parse_gt_file, macro_f_0_5, grid_search_thresholds, resolve_conflicts

DATA_DIR = os.environ.get('AMZ_ML_DATA_DIR', '../../dataset/train')
USE_DENSE = os.environ.get('AMZ_ML_USE_DENSE', '1') != '0'
MAX_CANDIDATES_PER_SOURCE = 30
NEG_PER_POS = 15
HARD_NEGATIVE_FRAC = 0.5  # fraction of sampled negatives biased toward highest-retrieval-score "near misses"
RANDOM_STATE = 42


def load_train_data(data_dir=DATA_DIR):
    df_s1 = pd.read_csv(os.path.join(data_dir, 'train_source1.tsv'), sep='\t')
    df_s2 = pd.read_csv(os.path.join(data_dir, 'train_source2.tsv'), sep='\t')
    df_s3 = pd.read_csv(os.path.join(data_dir, 'train_source3.tsv'), sep='\t')
    gt_df = pd.read_csv(os.path.join(data_dir, 'train_ground_truth.tsv'), sep='\t')
    return preprocess_df(df_s1), preprocess_df(df_s2), preprocess_df(df_s3), gt_df


def label_pairs(pair_scores_df, gt_dict):
    """Vectorized label assignment via merge, instead of a per-row Python set
    membership check -- matters once pair_scores_df is in the millions of rows."""
    gt_rows = [{'source1_entity_id': s1, 'candidate_entity_id': cid}
               for s1, matches in gt_dict.items() for cid in matches]
    gt_pairs_df = pd.DataFrame(gt_rows, columns=['source1_entity_id', 'candidate_entity_id'])
    gt_pairs_df['label'] = 1

    df = pair_scores_df.merge(gt_pairs_df, on=['source1_entity_id', 'candidate_entity_id'], how='left')
    df['label'] = df['label'].fillna(0).astype(int)
    df['target_source'] = np.where(df['candidate_entity_id'].astype(str).str.startswith('S2'), 1, 2)
    return df


def add_missing_gt_pairs(pairs_df, gt_dict):
    """Blocking can miss a true match entirely (it's outside anyone's top-k, or
    below the min-similarity floor). Those still need to appear in the *training*
    pool with label=1, or the model never sees an example of what that S1 entity's
    true match looks like. (They're handled separately, and correctly, as
    guaranteed false negatives in evaluate.build_scoring_table for the holdout
    metric -- this function is only about what the model trains on.)

    Score/rank columns for these injected rows are filled generically below
    rather than listed by name, since which enrichment columns exist depends on
    blocking.enrich_pair_scores' output."""
    covered = set(zip(pairs_df['source1_entity_id'], pairs_df['candidate_entity_id']))
    missing = []
    for s1, matches in gt_dict.items():
        for cid in matches:
            if (s1, cid) not in covered:
                missing.append({
                    'source1_entity_id': s1,
                    'candidate_entity_id': cid,
                    'label': 1,
                    'target_source': 1 if str(cid).startswith('S2') else 2,
                })
    if not missing:
        return pairs_df

    combined = pd.concat([pairs_df, pd.DataFrame(missing)], ignore_index=True)
    rank_cols = ['retrieval_rank_in_source', 'rank_for_candidate']
    for c in combined.columns:
        if c in ('source1_entity_id', 'candidate_entity_id', 'label', 'target_source'):
            continue
        combined[c] = combined[c].fillna(9999.0 if c in rank_cols else 0.0)
    return combined


def downsample_negatives(pairs_df, neg_per_pos=NEG_PER_POS, hard_frac=HARD_NEGATIVE_FRAC,
                          random_state=RANDOM_STATE):
    """
    Random negatives are almost all "obviously not a match" -- two businesses
    with nothing in common -- and teach the model little about the actual
    failure mode that hurts precision on F0.5: two DISTINCT but genuinely
    similar businesses (same chain in different cities, a franchise, a common
    name). Biasing the sampled negatives toward the highest-retrieval-score ones
    (the hardest, most confusable non-matches) directly targets that. The rest
    are still sampled randomly, both for diversity and so the model doesn't
    lose calibration on easy, obviously-negative cases.
    """
    pos = pairs_df[pairs_df['label'] == 1]
    neg = pairs_df[pairs_df['label'] == 0]
    max_neg = max(len(pos) * neg_per_pos, 1)
    if len(neg) <= max_neg:
        combined = pd.concat([pos, neg], ignore_index=True)
        return combined.sample(frac=1.0, random_state=random_state).reset_index(drop=True)

    if 'retrieval_score' in neg.columns:
        n_hard = int(max_neg * hard_frac)
        n_random = max_neg - n_hard
        hard_neg = neg.nlargest(n_hard, 'retrieval_score')
        remaining = neg.drop(hard_neg.index)
        random_neg = remaining.sample(n=min(n_random, len(remaining)), random_state=random_state)
        neg_sample = pd.concat([hard_neg, random_neg], ignore_index=True)
    else:
        neg_sample = neg.sample(n=max_neg, random_state=random_state)

    combined = pd.concat([pos, neg_sample], ignore_index=True)
    return combined.sample(frac=1.0, random_state=random_state).reset_index(drop=True)


def group_split(s1_ids, holdout_frac=0.15, val_frac=0.15, random_state=RANDOM_STATE):
    """Group split by source1_entity_id so no S1 entity's pairs leak across
    train/val/holdout -- the original code imported StratifiedGroupKFold for this
    exact purpose and then never called it; model.fit(X, y) ran on the full,
    ungrouped, undifferentiated set, so there was no held-out data to evaluate a
    threshold against at all. Two-stage GroupShuffleSplit: carve off the holdout
    first (used ONLY for threshold/cap tuning and the final reported F0.5, never
    seen during model fitting or calibration), then split the remainder into
    train/val (val is used for early stopping and calibration)."""
    s1_ids = np.array(sorted(set(s1_ids)))
    gss1 = GroupShuffleSplit(n_splits=1, test_size=holdout_frac, random_state=random_state)
    rest_idx, holdout_idx = next(gss1.split(s1_ids, groups=s1_ids))
    rest_ids, holdout_ids = s1_ids[rest_idx], s1_ids[holdout_idx]

    val_frac_of_rest = val_frac / (1 - holdout_frac)
    gss2 = GroupShuffleSplit(n_splits=1, test_size=val_frac_of_rest, random_state=random_state)
    train_idx, val_idx = next(gss2.split(rest_ids, groups=rest_ids))
    train_ids, val_ids = rest_ids[train_idx], rest_ids[val_idx]
    return set(train_ids), set(val_ids), set(holdout_ids)


def country_holdout_split(df_s1, holdout_country):
    """
    Diagnostic alternative to group_split: instead of a random group split (which
    still has every training country present in both train and holdout), hold
    out ONE ENTIRE COUNTRY. Training data only has US and India, but the test
    set adds France -- a random split never measures how well the model
    generalizes to an address/name style it has zero training examples of,
    because US and India are always present on both sides of that split.
    Training on India-only and evaluating on US-only (or vice versa) is the
    closest honest proxy available, from this data alone, for "how will this
    perform on France." Use via `train(country_holdout=<'US'|'india'>)`.

    This is a diagnostic, not the real training regime -- your actual submitted
    model should still train on all available countries (group_split), since
    throwing away a whole country's training signal only to estimate risk would
    be a bad trade for the real run.
    """
    country_col = df_s1['country_norm'].str.lower()
    train_ids = set(df_s1.loc[country_col != holdout_country.lower(), 'entity_id'])
    holdout_ids = set(df_s1.loc[country_col == holdout_country.lower(), 'entity_id'])
    # carve a small val slice out of train_ids for early stopping/calibration
    train_ids_arr = np.array(sorted(train_ids))
    gss = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=RANDOM_STATE)
    tr_idx, val_idx = next(gss.split(train_ids_arr, groups=train_ids_arr))
    return set(train_ids_arr[tr_idx]), set(train_ids_arr[val_idx]), holdout_ids


def train(data_dir=DATA_DIR, out_dir='.', country_holdout=None):
    """
    country_holdout: None for the real training run (random group split, uses
    all available countries). Pass e.g. "US" or "india" to instead run the
    leave-one-country-out diagnostic described in country_holdout_split's
    docstring -- prints the same diagnostics but does NOT overwrite
    model.pkl/thresholds.json, since it's deliberately trained on less data and
    is not meant to be your submitted model.
    """
    print("Loading data...")
    df_s1, df_s2, df_s3, gt_df = load_train_data(data_dir)
    gt_dict = parse_gt_file(gt_df)
    df_cand = pd.concat([df_s2, df_s3], ignore_index=True)

    print("Blocking...")
    be = BlockingEngine(use_dense=USE_DENSE)
    _, pair_scores_df, embeddings = be.fit_transform(df_s1, df_s2, df_s3)

    # Blocking recall ceiling -- the problem statement calls this out explicitly
    # ("invest in a strong blocking strategy -- it determines the upper bound of
    # your recall"). No amount of downstream model quality can recover a true
    # match blocking never retrieved, so this number bounds everything below it.
    retrieved = set(zip(pair_scores_df['source1_entity_id'], pair_scores_df['candidate_entity_id']))
    n_gt_pairs = sum(len(v) for v in gt_dict.values())
    n_gt_retrieved = sum(1 for s1, matches in gt_dict.items() for cid in matches if (s1, cid) in retrieved)
    recall_at_blocking = n_gt_retrieved / n_gt_pairs if n_gt_pairs else 1.0
    print(f"Blocking recall ceiling: {recall_at_blocking:.4f} ({n_gt_retrieved}/{n_gt_pairs} true matches retrieved)")

    pair_scores_df = cap_candidates_per_source(pair_scores_df, max_per_source=MAX_CANDIDATES_PER_SOURCE)

    retrieved_capped = set(zip(pair_scores_df['source1_entity_id'], pair_scores_df['candidate_entity_id']))
    n_gt_retrieved_capped = sum(1 for s1, matches in gt_dict.items() for cid in matches if (s1, cid) in retrieved_capped)
    recall_after_cap = n_gt_retrieved_capped / n_gt_pairs if n_gt_pairs else 1.0
    print(f"Recall after per-source cap ({MAX_CANDIDATES_PER_SOURCE}): {recall_after_cap:.4f}")
    if recall_after_cap < recall_at_blocking - 1e-9:
        print("  (cap is trimming some true matches -- raise MAX_CANDIDATES_PER_SOURCE if this gap matters)")

    # Rank/margin/mutual-nearest-neighbor columns computed ONCE here, on the
    # full post-cap pool, before any split or downsampling -- see
    # enrich_pair_scores' docstring for why order matters here.
    pair_scores_df = enrich_pair_scores(pair_scores_df)

    all_pairs = label_pairs(pair_scores_df, gt_dict)

    if country_holdout is None:
        train_ids, val_ids, holdout_ids = group_split(df_s1['entity_id'].tolist())
        print(f"Split (random, group by S1 entity): {len(train_ids)} train / {len(val_ids)} val / {len(holdout_ids)} holdout S1 entities")
    else:
        train_ids, val_ids, holdout_ids = country_holdout_split(df_s1, country_holdout)
        print(f"DIAGNOSTIC leave-one-country-out split (holdout country={country_holdout}): "
              f"{len(train_ids)} train / {len(val_ids)} val / {len(holdout_ids)} holdout S1 entities")

    train_pairs = add_missing_gt_pairs(
        all_pairs[all_pairs['source1_entity_id'].isin(train_ids)].copy(),
        {k: v for k, v in gt_dict.items() if k in train_ids})
    val_pairs = add_missing_gt_pairs(
        all_pairs[all_pairs['source1_entity_id'].isin(val_ids)].copy(),
        {k: v for k, v in gt_dict.items() if k in val_ids})
    # Holdout stays exactly as blocking produced it (post-cap) -- no downsampling,
    # no added missing-gt rows -- because it has to reflect the real deployment
    # distribution for the threshold search and final F0.5 to mean anything.
    holdout_pairs = all_pairs[all_pairs['source1_entity_id'].isin(holdout_ids)].copy()

    train_pairs = downsample_negatives(train_pairs)
    val_pairs = downsample_negatives(val_pairs)

    print(f"Train pairs: {len(train_pairs)} ({train_pairs['label'].mean():.3%} positive)")
    print(f"Val pairs: {len(val_pairs)} ({val_pairs['label'].mean():.3%} positive)")
    print(f"Holdout pairs (full candidate pool): {len(holdout_pairs)}")

    print("Extracting features...")
    fe = FeatureExtractor()
    train_feat = fe.get_features(train_pairs[['source1_entity_id', 'candidate_entity_id']], df_s1, df_cand,
                                  embeddings=embeddings, pair_scores_df=pair_scores_df)
    val_feat = fe.get_features(val_pairs[['source1_entity_id', 'candidate_entity_id']], df_s1, df_cand,
                                embeddings=embeddings, pair_scores_df=pair_scores_df)
    holdout_feat = fe.get_features(holdout_pairs[['source1_entity_id', 'candidate_entity_id']], df_s1, df_cand,
                                    embeddings=embeddings, pair_scores_df=pair_scores_df)

    feature_columns = [c for c in train_feat.columns if c not in ('source1_entity_id', 'candidate_entity_id')]

    train_feat = train_feat.merge(train_pairs[['source1_entity_id', 'candidate_entity_id', 'label']],
                                   on=['source1_entity_id', 'candidate_entity_id'], how='left')
    val_feat = val_feat.merge(val_pairs[['source1_entity_id', 'candidate_entity_id', 'label']],
                               on=['source1_entity_id', 'candidate_entity_id'], how='left')

    X_train = train_feat.reindex(columns=feature_columns, fill_value=0.0).values
    y_train = train_feat['label'].values
    X_val = val_feat.reindex(columns=feature_columns, fill_value=0.0).values
    y_val = val_feat['label'].values

    print("Training model (single pass per booster, early-stopped)...")
    model = EnsembleModel()
    model.fit(X_train, y_train, X_val, y_val, feature_names=feature_columns)

    print("Scoring holdout & tuning thresholds...")
    X_holdout = holdout_feat.reindex(columns=feature_columns, fill_value=0.0).values
    holdout_probs = model.predict_proba(X_holdout)[:, 1]

    holdout_scored = pd.DataFrame({
        'source1_entity_id': holdout_feat['source1_entity_id'].values,
        'candidate_entity_id': holdout_feat['candidate_entity_id'].values,
        'target_source': holdout_feat['target_source'].values,
        'prob': holdout_probs,
    })

    holdout_gt_dict = {k: v for k, v in gt_dict.items() if k in holdout_ids}
    t2, t3, cap, grid_score, _ = grid_search_thresholds(holdout_scored, holdout_gt_dict, holdout_ids)
    print(f"Best thresholds (independent per-S1 decisions): S2>={t2}, S3>={t3}, cap_per_source={cap} "
          f"-> grid-search F0.5={grid_score:.4f}")

    # Cross-check with the plain, obviously-correct reference implementation --
    # not just trusting the vectorized grid-search scorer -- using the SAME
    # independent-decision rule the grid search optimized.
    ranked = holdout_scored.copy()
    ranked['thresh'] = np.where(ranked['target_source'] == 1, t2, t3)
    ranked['rank'] = ranked.groupby(['source1_entity_id', 'target_source'])['prob'].rank(method='first', ascending=False)
    final_matches = ranked[(ranked['rank'] <= cap) & (ranked['prob'] >= ranked['thresh'])]
    pred_dict_independent = {s1: set(g['candidate_entity_id']) for s1, g in final_matches.groupby('source1_entity_id')}
    for s1 in holdout_ids:
        pred_dict_independent.setdefault(s1, set())
    f05_independent = macro_f_0_5(holdout_gt_dict, pred_dict_independent)
    print(f"Holdout macro F0.5, independent per-S1 decisions (reference implementation): {f05_independent:.4f}")

    # Global conflict resolution: S1 is the deduplicated reference set, so a
    # given S2/S3 record can belong to at most one S1 entity. This can only
    # remove wrong duplicate claims (never a correct one), so it should never
    # score below the independent-decision number above.
    pred_dict_resolved = resolve_conflicts(holdout_scored, t2, t3, cap)
    for s1 in holdout_ids:
        pred_dict_resolved.setdefault(s1, set())
    f05_resolved = macro_f_0_5(holdout_gt_dict, pred_dict_resolved)
    print(f"Holdout macro F0.5, WITH global conflict resolution: {f05_resolved:.4f} "
          f"({'+' if f05_resolved >= f05_independent else ''}{f05_resolved - f05_independent:+.4f} vs independent)")

    # Conflict resolution is a bet, not a guaranteed win (see resolve_conflicts'
    # docstring: it can lose if the model is more confident about a wrong claim
    # than a right one on a disputed candidate). Pick whichever wins on holdout
    # rather than assuming -- and persist that choice so pipeline.py applies the
    # same decision rule it was actually tuned/verified with.
    use_conflict_resolution = f05_resolved >= f05_independent
    final_f05 = f05_resolved if use_conflict_resolution else f05_independent
    print(f"-> Using {'conflict resolution' if use_conflict_resolution else 'independent decisions'} "
          f"for the real pipeline (higher holdout F0.5).")

    if country_holdout is not None:
        print(f"\n(Diagnostic run for holdout_country={country_holdout} -- NOT saving model.pkl/thresholds.json. "
              f"Compare this F0.5 to your real run's holdout F0.5 above: a large gap means the model may not "
              f"transfer well to France, an address style it has zero training examples of.)")
        return final_f05

    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, 'model.pkl'), 'wb') as f:
        pickle.dump({'model': model, 'feature_columns': feature_columns}, f)

    with open(os.path.join(out_dir, 'thresholds.json'), 'w') as f:
        json.dump({
            'threshold_s2': float(t2),
            'threshold_s3': float(t3),
            'cap_per_source': int(cap),
            'use_conflict_resolution': bool(use_conflict_resolution),
            'holdout_macro_f0_5_independent': float(f05_independent),
            'holdout_macro_f0_5_resolved': float(f05_resolved),
            'blocking_recall_ceiling': float(recall_at_blocking),
            'recall_after_cap': float(recall_after_cap),
            'max_candidates_per_source': MAX_CANDIDATES_PER_SOURCE,
        }, f, indent=2)

    print("Training complete. model.pkl and thresholds.json saved.")
    return final_f05


if __name__ == '__main__':
    # Deliberately no try/except here: the original code caught every exception,
    # printed a message, and then STILL wrote thresholds.json and printed
    # "Training complete" -- so a crash mid-training silently produced an
    # untrained model.pkl (or none at all) plus a confident-looking thresholds
    # file. Let failures surface.
    #
    # To run the leave-one-country-out diagnostic instead of the real training
    # run: python3 -c "from src.train import train; train(country_holdout='US')"
    train()
