import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm


class BlockingEngine:
    """
    Two-stage candidate retrieval:

      1) Character n-gram TF-IDF, blocked by normalized country (+ 'unknown' as a
         cross-country fallback on both sides), top-k via sklearn NearestNeighbors.
         NearestNeighbors(metric='cosine', algorithm='brute') computes similarity in
         chunks internally (sklearn.metrics.pairwise_distances_chunked) instead of
         materializing a dense (n_s1 x n_candidates) matrix. The original
         cosine_similarity() call built that matrix in full, which is a few hundred
         GB for a bucket with e.g. 100k x 500k rows -- an out-of-memory crash waiting
         to happen on any sizeable country group (especially 'unknown').

      2) Dense bi-encoder (MiniLM) similarity via FAISS, global (no country
         restriction), to catch semantic/paraphrase matches TF-IDF misses and to
         generalize to countries unseen in training.

    Returns:
      - candidates_df: source1_entity_id -> comma-joined candidate ids (for
        candidate_pairs.tsv, same shape the rest of the pipeline expects).
      - pair_scores_df: long-format (source1_entity_id, candidate_entity_id,
        tfidf_score, dense_score). Feeding these into FeatureExtractor means the
        similarity scores computed here are reused as model features instead of
        thrown away, and it's what the threshold/margin decision rule in train.py
        needs.
      - embeddings: dict entity_id -> dense embedding, so FeatureExtractor does not
        re-encode the same text a second time.
    """

    def __init__(self, use_dense=True, tfidf_top_k=25, dense_top_k=25,
                 tfidf_min_sim=0.05, dense_min_sim=0.5, dense_batch_size=256,
                 embedding_model_name='sentence-transformers/all-MiniLM-L6-v2',
                 ivf_threshold=200_000):
        self.use_dense = use_dense
        self.tfidf_top_k = tfidf_top_k
        self.dense_top_k = dense_top_k
        self.tfidf_min_sim = tfidf_min_sim
        self.dense_min_sim = dense_min_sim
        self.dense_batch_size = dense_batch_size
        self.embedding_model_name = embedding_model_name
        self.ivf_threshold = ivf_threshold
        self._st_model = None

    def _get_st_model(self):
        if self._st_model is None:
            from sentence_transformers import SentenceTransformer
            self._st_model = SentenceTransformer(self.embedding_model_name)
        return self._st_model

    @staticmethod
    def _country_col(df):
        if 'country_norm' in df.columns:
            return df['country_norm'].values
        return np.full(len(df), 'unknown', dtype=object)

    def _tfidf_stage(self, df_s1, candidates, s1_ids, cand_ids):
        country_s1 = self._country_col(df_s1)
        country_cand = self._country_col(candidates)

        s1_text = df_s1['combined_text'].values
        cand_text = candidates['combined_text'].values

        chunks = []
        for c in tqdm(pd.unique(country_s1), desc="TF-IDF Blocking"):
            c_s1_idx = np.where(country_s1 == c)[0]
            c_cand_idx = np.where((country_cand == c) | (country_cand == 'unknown'))[0]
            if len(c_s1_idx) == 0 or len(c_cand_idx) == 0:
                continue

            vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(3, 4), min_df=1)
            cand_tfidf = vectorizer.fit_transform(cand_text[c_cand_idx])
            s1_tfidf = vectorizer.transform(s1_text[c_s1_idx])

            k = min(self.tfidf_top_k, cand_tfidf.shape[0])
            nn = NearestNeighbors(n_neighbors=k, metric='cosine', algorithm='brute', n_jobs=-1)
            nn.fit(cand_tfidf)
            dist, idx = nn.kneighbors(s1_tfidf)
            sim = 1.0 - dist

            s1_id_flat = np.repeat(s1_ids[c_s1_idx], k)
            cand_id_flat = cand_ids[c_cand_idx[idx.ravel()]]
            sim_flat = sim.ravel()

            keep = sim_flat > self.tfidf_min_sim
            if keep.any():
                chunks.append(pd.DataFrame({
                    'source1_entity_id': s1_id_flat[keep],
                    'candidate_entity_id': cand_id_flat[keep],
                    'tfidf_score': sim_flat[keep],
                }))

        if not chunks:
            return pd.DataFrame(columns=['source1_entity_id', 'candidate_entity_id', 'tfidf_score'])
        return pd.concat(chunks, ignore_index=True)

    def _dense_stage(self, df_s1, candidates, s1_ids, cand_ids):
        import faiss

        model = self._get_st_model()
        cand_emb = np.asarray(model.encode(candidates['combined_text'].tolist(),
                                            batch_size=self.dense_batch_size,
                                            show_progress_bar=True), dtype='float32')
        s1_emb = np.asarray(model.encode(df_s1['combined_text'].tolist(),
                                          batch_size=self.dense_batch_size,
                                          show_progress_bar=True), dtype='float32')
        faiss.normalize_L2(cand_emb)
        faiss.normalize_L2(s1_emb)

        embeddings = {eid: emb for eid, emb in zip(cand_ids, cand_emb)}
        embeddings.update({eid: emb for eid, emb in zip(s1_ids, s1_emb)})

        d = cand_emb.shape[1]
        n_cand = cand_emb.shape[0]
        if n_cand > self.ivf_threshold:
            # Exact FlatIP search is O(n_queries * n_candidates); past a few hundred
            # thousand candidates that gets slow. IVF trades a little recall for a
            # large speedup -- nlist ~ sqrt(n) is the standard rule of thumb.
            nlist = max(64, int(np.sqrt(n_cand)))
            quantizer = faiss.IndexFlatIP(d)
            index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_INNER_PRODUCT)
            index.train(cand_emb)
            index.nprobe = min(32, nlist)
        else:
            index = faiss.IndexFlatIP(d)
        index.add(cand_emb)

        k = min(self.dense_top_k, n_cand)
        D, I = index.search(s1_emb, k)

        s1_id_flat = np.repeat(s1_ids, k)
        valid = I.ravel() != -1
        cand_id_flat = np.where(valid, cand_ids[np.clip(I.ravel(), 0, len(cand_ids) - 1)], "")
        score_flat = D.ravel()

        keep = valid & (score_flat > self.dense_min_sim)
        dense_df = pd.DataFrame({
            'source1_entity_id': s1_id_flat[keep],
            'candidate_entity_id': cand_id_flat[keep],
            'dense_score': score_flat[keep],
        })
        return dense_df, embeddings

    def fit_transform(self, df_s1, df_s2, df_s3):
        df_s1 = df_s1.copy()
        candidates = pd.concat([df_s2, df_s3], ignore_index=True)

        df_s1['combined_text'] = (df_s1['clean_name'].fillna('') + " " + df_s1['clean_address'].fillna('')).str.strip()
        candidates['combined_text'] = (candidates['clean_name'].fillna('') + " " + candidates['clean_address'].fillna('')).str.strip()

        s1_ids = df_s1['entity_id'].values
        cand_ids = candidates['entity_id'].values

        tfidf_df = self._tfidf_stage(df_s1, candidates, s1_ids, cand_ids)

        embeddings = {}
        if self.use_dense:
            dense_df, embeddings = self._dense_stage(df_s1, candidates, s1_ids, cand_ids)
        else:
            dense_df = pd.DataFrame(columns=['source1_entity_id', 'candidate_entity_id', 'dense_score'])

        pair_scores_df = pd.merge(tfidf_df, dense_df, on=['source1_entity_id', 'candidate_entity_id'], how='outer')
        pair_scores_df['tfidf_score'] = pair_scores_df.get('tfidf_score', 0.0)
        pair_scores_df['dense_score'] = pair_scores_df.get('dense_score', 0.0)
        pair_scores_df[['tfidf_score', 'dense_score']] = pair_scores_df[['tfidf_score', 'dense_score']].fillna(0.0).astype('float64')

        grouped = pair_scores_df.groupby('source1_entity_id')['candidate_entity_id'].apply(
            lambda s: ",".join(sorted(s.unique()))
        )
        candidates_out_df = grouped.reindex(s1_ids, fill_value="").reset_index()
        candidates_out_df.columns = ['source1_entity_id', 'candidate_entity_ids']

        return candidates_out_df, pair_scores_df, embeddings


def cap_candidates_per_source(pair_scores_df, max_per_source=30):
    """
    Keep only the top `max_per_source` candidates (by max(tfidf_score, dense_score))
    per (source1_entity_id, source) group. Blocking with top_k=25 per stage per s1
    can still produce a long tail once TF-IDF and dense hits are unioned; running
    feature extraction and model inference on all of them is wasted compute on
    candidates that were never going to be picked (this is what actually gets fed
    to the model and therefore what belongs in candidate_pairs.tsv per the problem
    statement -- "the final candidate list just before the ML model scores them").

    Keep max_per_source comfortably above the largest cap you grid-search over in
    evaluate.grid_search_thresholds, or you'll silently cut off candidates the
    decision rule could otherwise have used.
    """
    if pair_scores_df.empty:
        return pair_scores_df
    df = pair_scores_df.copy()
    df['target_source'] = np.where(df['candidate_entity_id'].astype(str).str.startswith('S2'), 1, 2)
    df['retrieval_score'] = np.maximum(df['tfidf_score'], df['dense_score'])
    df['_rank'] = df.groupby(['source1_entity_id', 'target_source'])['retrieval_score'] \
        .rank(method='first', ascending=False)
    kept = df[df['_rank'] <= max_per_source].drop(columns=['_rank', 'target_source', 'retrieval_score'])
    return kept.reset_index(drop=True)
