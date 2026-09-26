import pandas as pd
import json
import pickle
import os
import sys
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.preprocess import preprocess_df
from src.blocking import BlockingEngine, cap_candidates_per_source
from src.features import FeatureExtractor

DATA_DIR = os.environ.get('AMZ_ML_TEST_DIR', '../../dataset/test')
OUT_DIR = os.environ.get('AMZ_ML_OUT_DIR', '../../output')
MAX_CANDIDATES_PER_SOURCE = int(os.environ.get('AMZ_ML_MAX_CAND_PER_SOURCE', '30'))
USE_DENSE = os.environ.get('AMZ_ML_USE_DENSE', '1') != '0'


def run_pipeline():
    df_s1 = pd.read_csv(os.path.join(DATA_DIR, 'test_source1.tsv'), sep='\t')
    df_s2 = pd.read_csv(os.path.join(DATA_DIR, 'test_source2.tsv'), sep='\t')
    df_s3 = pd.read_csv(os.path.join(DATA_DIR, 'test_source3.tsv'), sep='\t')

    df_s1 = preprocess_df(df_s1)
    df_s2 = preprocess_df(df_s2)
    df_s3 = preprocess_df(df_s3)
    df_cand = pd.concat([df_s2, df_s3], ignore_index=True)

    be = BlockingEngine(use_dense=USE_DENSE)
    _, pair_scores_df, embeddings = be.fit_transform(df_s1, df_s2, df_s3)

    # Cap to the same per-source limit used in training, BEFORE writing
    # candidate_pairs.tsv -- the problem statement is explicit that this file must
    # be "the final candidate list just before the ML model scores them," not the
    # raw union straight out of blocking.
    pair_scores_df = cap_candidates_per_source(pair_scores_df, max_per_source=MAX_CANDIDATES_PER_SOURCE)

    os.makedirs(OUT_DIR, exist_ok=True)
    all_s1_ids = df_s1['entity_id'].tolist()

    if pair_scores_df.empty:
        cand_out = pd.DataFrame({'source1_entity_id': all_s1_ids, 'candidate_entity_ids': ""})
        cand_out.to_csv(os.path.join(OUT_DIR, 'candidate_pairs.tsv'), sep='\t', index=False)
        match_out = pd.DataFrame({'source1_entity_id': all_s1_ids, 'matched_entity_ids': ""})
        match_out.to_csv(os.path.join(OUT_DIR, 'matching_results.tsv'), sep='\t', index=False)
        print("No candidates generated. Wrote empty-match output for all S1 entities.")
        return

    cand_grouped = pair_scores_df.groupby('source1_entity_id')['candidate_entity_id'] \
        .apply(lambda s: ",".join(sorted(s.unique())))
    cand_out = cand_grouped.reindex(all_s1_ids, fill_value="").reset_index()
    cand_out.columns = ['source1_entity_id', 'candidate_entity_ids']
    cand_out.to_csv(os.path.join(OUT_DIR, 'candidate_pairs.tsv'), sep='\t', index=False)

    fe = FeatureExtractor()
    pairs_df = pair_scores_df[['source1_entity_id', 'candidate_entity_id']].copy()
    features_df = fe.get_features(pairs_df, df_s1, df_cand, embeddings=embeddings, pair_scores_df=pair_scores_df)

    model_path = os.path.join(os.path.dirname(__file__), '..', 'model.pkl')
    thresholds_path = os.path.join(os.path.dirname(__file__), '..', 'thresholds.json')

    try:
        with open(model_path, 'rb') as f:
            bundle = pickle.load(f)
        model, feature_columns = bundle['model'], bundle['feature_columns']
        with open(thresholds_path, 'r') as f:
            th = json.load(f)
    except FileNotFoundError as e:
        # Fail loudly rather than silently emitting an all-empty submission --
        # a missing model.pkl/thresholds.json means train.py hasn't been run
        # (or crashed) and that needs to be noticed, not masked as "0 matches".
        raise RuntimeError(
            f"Could not load model/thresholds ({e}). Run train.py before pipeline.py."
        ) from e

    threshold_s2 = th['threshold_s2']
    threshold_s3 = th['threshold_s3']
    cap = th['cap_per_source']

    X = features_df.reindex(columns=feature_columns, fill_value=0.0).values
    probs = model.predict_proba(X)[:, 1]

    scored = pd.DataFrame({
        'source1_entity_id': features_df['source1_entity_id'].values,
        'candidate_entity_id': features_df['candidate_entity_id'].values,
        'target_source': features_df['target_source'].values,
        'prob': probs,
    })
    scored['thresh'] = np.where(scored['target_source'] == 1, threshold_s2, threshold_s3)
    scored['rank'] = scored.groupby(['source1_entity_id', 'target_source'])['prob'] \
        .rank(method='first', ascending=False)

    final_matches = scored[(scored['rank'] <= cap) & (scored['prob'] >= scored['thresh'])]
    matched_grouped = final_matches.groupby('source1_entity_id')['candidate_entity_id'] \
        .apply(lambda s: ",".join(sorted(s.unique())))
    match_out = matched_grouped.reindex(all_s1_ids, fill_value="").reset_index()
    match_out.columns = ['source1_entity_id', 'matched_entity_ids']
    match_out.to_csv(os.path.join(OUT_DIR, 'matching_results.tsv'), sep='\t', index=False)

    n_matched = (match_out['matched_entity_ids'] != "").sum()
    print(f"Pipeline complete. {n_matched}/{len(all_s1_ids)} S1 entities matched. "
          f"Output written to {OUT_DIR}/matching_results.tsv")


if __name__ == '__main__':
    run_pipeline()
