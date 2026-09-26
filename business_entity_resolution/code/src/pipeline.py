import pandas as pd
import json
import pickle
import os
import sys
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.preprocess import preprocess_df
from src.blocking import BlockingEngine
from src.features import FeatureExtractor

def run_pipeline():
    df_s1 = pd.read_csv('../../dataset/test/test_source1.tsv', sep='\t')
    df_s2 = pd.read_csv('../../dataset/test/test_source2.tsv', sep='\t')
    df_s3 = pd.read_csv('../../dataset/test/test_source3.tsv', sep='\t')
    
    df_s1 = preprocess_df(df_s1)
    df_s2 = preprocess_df(df_s2)
    df_s3 = preprocess_df(df_s3)
    
    be = BlockingEngine(use_dense=True)
    cands_df = be.fit_transform(df_s1, df_s2, df_s3)
    
    os.makedirs('../../output', exist_ok=True)
    cands_df.to_csv('../../output/candidate_pairs.tsv', sep='\t', index=False)
    
    pairs = []
    for _, row in cands_df.iterrows():
        s1 = row['source1_entity_id']
        if pd.notna(row['candidate_entity_ids']):
            cands = str(row['candidate_entity_ids']).split(',')
            for c in cands:
                if c:
                    pairs.append({'source1_entity_id': s1, 'candidate_entity_id': c})
                
    pairs_df = pd.DataFrame(pairs)
    if len(pairs_df) == 0:
        with open('../../output/matching_results.tsv', 'w') as f:
            f.write("source1_entity_id\tmatched_entity_ids\n")
            for _, row in df_s1.iterrows():
                f.write(f"{row['entity_id']}\t\n")
        return
        
    df_cand = pd.concat([df_s2, df_s3], ignore_index=True)
    fe = FeatureExtractor()
    features_df = fe.get_features(pairs_df, df_s1, df_cand)
    
    X = features_df.drop(columns=['source1_entity_id', 'candidate_entity_id'])
    
    try:
        with open('model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('thresholds.json', 'r') as f:
            th = json.load(f)
            threshold = th['threshold']
            singleton_th = th['singleton_threshold']
    except Exception as e:
        print(f"Loading model failed: {e}. Using dummy predictions.")
        threshold = 0.5
        singleton_th = 0.5
        model = None
        
    if model is not None and not X.empty:
        probs = model.predict_proba(X)[:, 1]
    else:
        probs = np.zeros(len(X))
        
    pairs_df['prob'] = probs
    
    matching_results = []
    for s1_id, group in pairs_df.groupby('source1_entity_id'):
        max_prob = group['prob'].max()
        if max_prob < singleton_th:
            matching_results.append({'source1_entity_id': s1_id, 'matched_entity_ids': ''})
        else:
            matches = group[group['prob'] >= threshold]['candidate_entity_id'].tolist()
            if matches:
                matching_results.append({'source1_entity_id': s1_id, 'matched_entity_ids': ",".join(matches)})
            else:
                matching_results.append({'source1_entity_id': s1_id, 'matched_entity_ids': ''})
            
    s1_matched = set(x['source1_entity_id'] for x in matching_results)
    for s1_id in df_s1['entity_id']:
        if s1_id not in s1_matched:
            matching_results.append({'source1_entity_id': s1_id, 'matched_entity_ids': ''})
            
    out_df = pd.DataFrame(matching_results)
    out_df.to_csv('../../output/matching_results.tsv', sep='\t', index=False)
    print("Pipeline complete. Output written to output/matching_results.tsv")

if __name__ == '__main__':
    run_pipeline()
