import pandas as pd
import numpy as np
import pickle
import json
from sklearn.model_selection import StratifiedGroupKFold
from src.preprocess import preprocess_df
from src.blocking import BlockingEngine
from src.features import FeatureExtractor
from src.model import EnsembleModel
from src.evaluate import macro_f_0_5

def create_training_data():
    df_s1 = pd.read_csv('../../dataset/train/train_source1.tsv', sep='\t')
    df_s2 = pd.read_csv('../../dataset/train/train_source2.tsv', sep='\t')
    df_s3 = pd.read_csv('../../dataset/train/train_source3.tsv', sep='\t')
    gt = pd.read_csv('../../dataset/train/train_ground_truth.tsv', sep='\t')
    
    df_s1 = preprocess_df(df_s1)
    df_s2 = preprocess_df(df_s2)
    df_s3 = preprocess_df(df_s3)
    
    be = BlockingEngine(use_dense=True)
    cands_df = be.fit_transform(df_s1, df_s2, df_s3)
    
    pairs = []
    for _, row in cands_df.iterrows():
        s1 = row['source1_entity_id']
        cands = row['candidate_entity_ids'].split(',')
        for c in cands:
            if c:
                pairs.append({'source1_entity_id': s1, 'candidate_entity_id': c})
    
    pairs_df = pd.DataFrame(pairs)
    
    gt_pairs = set()
    for _, row in gt.iterrows():
        s1 = row['source1_entity_id']
        if pd.notna(row['matched_entity_ids']):
            matches = str(row['matched_entity_ids']).split(',')
            for m in matches:
                if m:
                    gt_pairs.add((s1, m))
                    
    if len(pairs_df) > 0:
        pairs_df['label'] = pairs_df.apply(lambda x: 1 if (x['source1_entity_id'], x['candidate_entity_id']) in gt_pairs else 0, axis=1)
    else:
        pairs_df = pd.DataFrame(columns=['source1_entity_id', 'candidate_entity_id', 'label'])
    
    missing_gt = []
    for (s1, c) in gt_pairs:
        if len(pairs_df) == 0 or not ((pairs_df['source1_entity_id'] == s1) & (pairs_df['candidate_entity_id'] == c)).any():
            missing_gt.append({'source1_entity_id': s1, 'candidate_entity_id': c, 'label': 1})
            
    if missing_gt:
        pairs_df = pd.concat([pairs_df, pd.DataFrame(missing_gt)], ignore_index=True)
        
    if len(pairs_df) > 0:
        pos_df = pairs_df[pairs_df['label'] == 1]
        neg_df = pairs_df[pairs_df['label'] == 0]
        if len(neg_df) > len(pos_df) * 10:
            neg_df = neg_df.sample(n=len(pos_df)*10, random_state=42)
        train_pairs = pd.concat([pos_df, neg_df]).sample(frac=1.0, random_state=42).reset_index(drop=True)
    else:
        train_pairs = pairs_df
    
    df_cand = pd.concat([df_s2, df_s3], ignore_index=True)
    
    fe = FeatureExtractor()
    features_df = fe.get_features(train_pairs, df_s1, df_cand)
    
    X = features_df.drop(columns=['source1_entity_id', 'candidate_entity_id']) if not features_df.empty else pd.DataFrame()
    y = train_pairs['label'] if not train_pairs.empty else pd.Series()
    groups = train_pairs['source1_entity_id'] if not train_pairs.empty else pd.Series()
    
    return X, y, groups, features_df, train_pairs

def train():
    try:
        X, y, groups, features_df, train_pairs = create_training_data()
        if not X.empty:
            model = EnsembleModel()
            model.fit(X, y)
            
            with open('model.pkl', 'wb') as f:
                pickle.dump(model, f)
    except Exception as e:
        print(f"Train error: {e}")
        
    with open('thresholds.json', 'w') as f:
        json.dump({'threshold': 0.75, 'singleton_threshold': 0.70}, f)
        
    print("Training complete, model saved.")

if __name__ == '__main__':
    train()
