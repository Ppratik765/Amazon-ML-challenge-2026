import pandas as pd
import numpy as np
import Levenshtein
from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cosine
from tqdm import tqdm

def jaccard_set(s1, s2):
    if not s1 and not s2:
        return 1.0
    if not s1 or not s2:
        return 0.0
    return len(s1.intersection(s2)) / len(s1.union(s2))

def token_sort_ratio(s1, s2):
    t1 = " ".join(sorted(str(s1).split()))
    t2 = " ".join(sorted(str(s2).split()))
    return Levenshtein.ratio(t1, t2)

def token_set_ratio(s1, s2):
    set1 = set(str(s1).split())
    set2 = set(str(s2).split())
    intersection = set1.intersection(set2)
    diff1 = set1 - intersection
    diff2 = set2 - intersection
    
    t_inter = " ".join(sorted(intersection))
    t1 = " ".join(sorted(intersection.union(diff1)))
    t2 = " ".join(sorted(intersection.union(diff2)))
    
    return max([
        Levenshtein.ratio(t_inter, t1),
        Levenshtein.ratio(t_inter, t2),
        Levenshtein.ratio(t1, t2)
    ])

def first_token_match(s1, s2):
    t1 = str(s1).split()
    t2 = str(s2).split()
    if not t1 or not t2:
        return 0
    return 1 if t1[0] == t2[0] else 0

def acronym_match(s1, s2):
    t1 = "".join([w[0] for w in str(s1).split() if w])
    t2 = "".join([w[0] for w in str(s2).split() if w])
    if t1 and t2 and (t1 == str(s2) or t2 == str(s1)):
        return 1
    return 0
    
class FeatureExtractor:
    def __init__(self):
        self.model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        
    def get_features(self, df_pairs, df_s1, df_cand):
        df_s1_idx = df_s1.set_index('entity_id')
        df_cand_idx = df_cand.set_index('entity_id')
        
        s1_names = df_s1['clean_name'].fillna('').tolist()
        s1_addrs = df_s1['clean_address'].fillna('').tolist()
        cand_names = df_cand['clean_name'].fillna('').tolist()
        cand_addrs = df_cand['clean_address'].fillna('').tolist()
        
        s1_name_emb = {id_: emb for id_, emb in zip(df_s1['entity_id'], self.model.encode(s1_names, batch_size=256))}
        s1_addr_emb = {id_: emb for id_, emb in zip(df_s1['entity_id'], self.model.encode(s1_addrs, batch_size=256))}
        cand_name_emb = {id_: emb for id_, emb in zip(df_cand['entity_id'], self.model.encode(cand_names, batch_size=256))}
        cand_addr_emb = {id_: emb for id_, emb in zip(df_cand['entity_id'], self.model.encode(cand_addrs, batch_size=256))}
        
        features = []
        for _, row in tqdm(df_pairs.iterrows(), total=len(df_pairs), desc="Extracting Features"):
            s1_id = row['source1_entity_id']
            c_id = row['candidate_entity_id']
            
            s1_row = df_s1_idx.loc[s1_id]
            c_row = df_cand_idx.loc[c_id]
            
            n1 = str(s1_row.get('clean_name', ''))
            n2 = str(c_row.get('clean_name', ''))
            a1 = str(s1_row.get('clean_address', ''))
            a2 = str(c_row.get('clean_address', ''))
            
            feat = {
                'source1_entity_id': s1_id,
                'candidate_entity_id': c_id,
                'name_lev_dist': Levenshtein.distance(n1, n2),
                'name_lev_ratio': Levenshtein.ratio(n1, n2),
                'name_jaro_winkler': Levenshtein.jaro_winkler(n1, n2),
                'name_token_sort': token_sort_ratio(n1, n2),
                'name_token_set': token_set_ratio(n1, n2),
                'addr_lev_dist': Levenshtein.distance(a1, a2),
                'addr_lev_ratio': Levenshtein.ratio(a1, a2),
                'addr_jaro_winkler': Levenshtein.jaro_winkler(a1, a2),
                'name_exact_match': int(n1 == n2 and n1 != ""),
                'addr_exact_match': int(a1 == a2 and a1 != ""),
                'name_first_token': first_token_match(n1, n2),
                'name_acronym_match': acronym_match(n1, n2),
            }
            
            nums1_n = s1_row.get('nums_name', set())
            nums2_n = c_row.get('nums_name', set())
            nums1_a = s1_row.get('nums_address', set())
            nums2_a = c_row.get('nums_address', set())
            
            feat['nums_name_jaccard'] = jaccard_set(nums1_n, nums2_n)
            feat['nums_addr_jaccard'] = jaccard_set(nums1_a, nums2_a)
            
            feat['nums_overlap_count'] = len(nums1_n.intersection(nums2_n)) + len(nums1_a.intersection(nums2_a))
            feat['nums_conflict_flag'] = int((len(nums1_a) > 0 and len(nums2_a) > 0 and feat['nums_addr_jaccard'] < 1.0))
            
            emb_n1 = s1_name_emb[s1_id]
            emb_n2 = cand_name_emb[c_id]
            emb_a1 = s1_addr_emb[s1_id]
            emb_a2 = cand_addr_emb[c_id]
            
            feat['name_emb_cos'] = 1 - cosine(emb_n1, emb_n2) if np.any(emb_n1) and np.any(emb_n2) else 0.0
            feat['addr_emb_cos'] = 1 - cosine(emb_a1, emb_a2) if np.any(emb_a1) and np.any(emb_a2) else 0.0
            
            feat['target_source'] = 1 if c_id.startswith('S2') else 2
            feat['name_len_diff_ratio'] = abs(len(n1) - len(n2)) / max(1, max(len(n1), len(n2)))
            feat['addr_len_diff_ratio'] = abs(len(a1) - len(a2)) / max(1, max(len(a1), len(a2)))
            
            features.append(feat)
            
        return pd.DataFrame(features)
