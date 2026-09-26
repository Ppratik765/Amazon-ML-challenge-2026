import pandas as pd
import numpy as np
import faiss
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

class BlockingEngine:
    def __init__(self, use_dense=True):
        self.use_dense = use_dense
        if use_dense:
            self.model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    
    def fit_transform(self, df_s1, df_s2, df_s3):
        candidates = pd.concat([df_s2, df_s3], ignore_index=True)
        
        candidates['combined_text'] = candidates['clean_name'] + " " + candidates['clean_address']
        df_s1['combined_text'] = df_s1['clean_name'] + " " + df_s1['clean_address']
        
        candidates['combined_text'] = candidates['combined_text'].fillna('')
        df_s1['combined_text'] = df_s1['combined_text'].fillna('')
        
        s1_ids = df_s1['entity_id'].values
        cand_ids = candidates['entity_id'].values
        
        candidate_dict = {s1_id: set() for s1_id in s1_ids}
        
        df_s1['country_filled'] = df_s1['country'].fillna('unknown').astype(str).str.lower()
        candidates['country_filled'] = candidates['country'].fillna('unknown').astype(str).str.lower()
        
        for c in tqdm(df_s1['country_filled'].unique(), desc="TF-IDF Blocking"):
            c_s1_idx = df_s1.index[df_s1['country_filled'] == c].tolist()
            c_cand_idx = candidates.index[(candidates['country_filled'] == c) | (candidates['country_filled'] == 'unknown')].tolist()
            
            if len(c_cand_idx) == 0 or len(c_s1_idx) == 0:
                continue
            
            vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(3,4))
            cand_tfidf = vectorizer.fit_transform(candidates.iloc[c_cand_idx]['combined_text'])
            s1_tfidf = vectorizer.transform(df_s1.iloc[c_s1_idx]['combined_text'])
            
            from sklearn.metrics.pairwise import cosine_similarity
            sim_matrix = cosine_similarity(s1_tfidf, cand_tfidf)
            
            for i, s1_i in enumerate(c_s1_idx):
                top_25_idx = np.argsort(sim_matrix[i])[-25:]
                for cand_i_mapped in top_25_idx:
                    cand_i = c_cand_idx[cand_i_mapped]
                    if sim_matrix[i, cand_i_mapped] > 0.05: 
                        candidate_dict[s1_ids[s1_i]].add(cand_ids[cand_i])
                        
        if self.use_dense:
            cand_embeddings = self.model.encode(candidates['combined_text'].tolist(), batch_size=256, show_progress_bar=True)
            s1_embeddings = self.model.encode(df_s1['combined_text'].tolist(), batch_size=256, show_progress_bar=True)
            
            d = cand_embeddings.shape[1]
            index = faiss.IndexFlatIP(d)
            faiss.normalize_L2(cand_embeddings)
            faiss.normalize_L2(s1_embeddings)
            index.add(cand_embeddings)
            
            D, I = index.search(s1_embeddings, 25)
            for i, s1_id in enumerate(s1_ids):
                for j, cand_i in enumerate(I[i]):
                    if cand_i != -1 and D[i][j] > 0.5:
                        candidate_dict[s1_id].add(cand_ids[cand_i])
        
        res = []
        for s1_id in s1_ids:
            cands = sorted(list(candidate_dict[s1_id]))
            res.append({
                'source1_entity_id': s1_id,
                'candidate_entity_ids': ",".join(cands)
            })
        
        return pd.DataFrame(res)
