import pandas as pd
import numpy as np
from rapidfuzz import fuzz
from rapidfuzz.distance import Levenshtein, JaroWinkler
from tqdm import tqdm

tqdm.pandas()


def _jaccard_row(pair):
    s1, s2 = pair
    if not s1 and not s2:
        return 1.0
    if not s1 or not s2:
        return 0.0
    return len(s1 & s2) / len(s1 | s2)


def _acronym(text):
    return "".join(w[0] for w in str(text).split() if w)


class FeatureExtractor:
    """
    Computes pairwise similarity features for (source1, candidate) pairs.

    Fully vectorized: the original implementation looped over df_pairs.iterrows()
    and did per-row dict lookups for every feature, which does not scale past a few
    tens of thousands of pairs. This version merges S1/candidate attributes into the
    pairs frame once, then computes every feature as a column operation (rapidfuzz
    calls are still one-per-pair under the hood, but that's an unavoidable C-level
    string comparison, not the Python-level overhead the old row loop added on top).

    Accepts optional `embeddings` (entity_id -> vector) and `pair_scores_df`
    (source1_entity_id, candidate_entity_id, tfidf_score, dense_score) produced by
    BlockingEngine, so text does not get re-encoded a second time here.
    """

    def __init__(self, embedding_model_name='sentence-transformers/all-MiniLM-L6-v2'):
        self.embedding_model_name = embedding_model_name
        self._st_model = None

    def _get_st_model(self):
        if self._st_model is None:
            from sentence_transformers import SentenceTransformer
            self._st_model = SentenceTransformer(self.embedding_model_name)
        return self._st_model

    def _ensure_embeddings(self, df_s1, df_cand, embeddings):
        """Fallback path: compute name/address embeddings if the caller didn't pass
        ones already computed during blocking. Prefer passing them in -- encoding
        the same text twice (once in BlockingEngine, once here) wastes real time at
        scale for zero benefit."""
        if embeddings is not None:
            return embeddings
        model = self._get_st_model()
        name_texts = pd.concat([df_s1['clean_name'], df_cand['clean_name']]).fillna('').tolist()
        addr_texts = pd.concat([df_s1['clean_address'], df_cand['clean_address']]).fillna('').tolist()
        all_ids = pd.concat([df_s1['entity_id'], df_cand['entity_id']]).tolist()
        name_emb = model.encode(name_texts, batch_size=256, show_progress_bar=True)
        addr_emb = model.encode(addr_texts, batch_size=256, show_progress_bar=True)
        emb = {}
        for i, eid in enumerate(all_ids):
            v = np.concatenate([name_emb[i], addr_emb[i]])
            emb[eid] = v
        return emb

    def get_features(self, df_pairs, df_s1, df_cand, embeddings=None, pair_scores_df=None):
        if df_pairs.empty:
            return pd.DataFrame(columns=['source1_entity_id', 'candidate_entity_id'])

        s1_cols = ['entity_id', 'clean_name', 'clean_address', 'nums_name', 'nums_address',
                   'street_num', 'postal_code', 'country_norm']
        cand_cols = s1_cols
        s1_small = df_s1[[c for c in s1_cols if c in df_s1.columns]].add_prefix('s1_')
        cand_small = df_cand[[c for c in cand_cols if c in df_cand.columns]].add_prefix('c_')

        df = df_pairs.merge(s1_small, left_on='source1_entity_id', right_on='s1_entity_id', how='left')
        df = df.merge(cand_small, left_on='candidate_entity_id', right_on='c_entity_id', how='left')

        n1 = df['s1_clean_name'].fillna('').astype(str)
        n2 = df['c_clean_name'].fillna('').astype(str)
        a1 = df['s1_clean_address'].fillna('').astype(str)
        a2 = df['c_clean_address'].fillna('').astype(str)

        feat = pd.DataFrame(index=df.index)
        feat['source1_entity_id'] = df['source1_entity_id']
        feat['candidate_entity_id'] = df['candidate_entity_id']

        # --- string similarity (rapidfuzz; C-implemented, much faster than the
        # original pure-python Levenshtein + hand-rolled token_sort/token_set) ---
        feat['name_lev_dist'] = [Levenshtein.distance(x, y) for x, y in zip(n1, n2)]
        feat['name_lev_ratio'] = [fuzz.ratio(x, y) / 100.0 for x, y in zip(n1, n2)]
        feat['name_jaro_winkler'] = [JaroWinkler.normalized_similarity(x, y) for x, y in zip(n1, n2)]
        feat['name_token_sort'] = [fuzz.token_sort_ratio(x, y) / 100.0 for x, y in zip(n1, n2)]
        feat['name_token_set'] = [fuzz.token_set_ratio(x, y) / 100.0 for x, y in zip(n1, n2)]

        feat['addr_lev_dist'] = [Levenshtein.distance(x, y) for x, y in zip(a1, a2)]
        feat['addr_lev_ratio'] = [fuzz.ratio(x, y) / 100.0 for x, y in zip(a1, a2)]
        feat['addr_jaro_winkler'] = [JaroWinkler.normalized_similarity(x, y) for x, y in zip(a1, a2)]
        feat['addr_token_sort'] = [fuzz.token_sort_ratio(x, y) / 100.0 for x, y in zip(a1, a2)]

        feat['name_exact_match'] = ((n1 == n2) & (n1 != "")).astype(int)
        feat['addr_exact_match'] = ((a1 == a2) & (a1 != "")).astype(int)

        n1_first = n1.str.split().str[0].fillna('')
        n2_first = n2.str.split().str[0].fillna('')
        feat['name_first_token'] = (n1_first == n2_first).astype(int)

        n1_acr = n1.apply(_acronym)
        n2_acr = n2.apply(_acronym)
        feat['name_acronym_match'] = (((n1_acr == n2) & (n1_acr != "")) | ((n2_acr == n1) & (n2_acr != ""))).astype(int)

        # --- numeric / structured fields ---
        if 's1_nums_name' in df.columns:
            empty_fs = frozenset()
            s1_nn = df['s1_nums_name'].apply(lambda x: x if isinstance(x, frozenset) else empty_fs)
            c_nn = df['c_nums_name'].apply(lambda x: x if isinstance(x, frozenset) else empty_fs)
            s1_na = df['s1_nums_address'].apply(lambda x: x if isinstance(x, frozenset) else empty_fs)
            c_na = df['c_nums_address'].apply(lambda x: x if isinstance(x, frozenset) else empty_fs)
            feat['nums_name_jaccard'] = list(map(_jaccard_row, zip(s1_nn, c_nn)))
            feat['nums_addr_jaccard'] = list(map(_jaccard_row, zip(s1_na, c_na)))
            overlap_name = [len(a & b) for a, b in zip(s1_nn, c_nn)]
            overlap_addr = [len(a & b) for a, b in zip(s1_na, c_na)]
            feat['nums_overlap_count'] = np.array(overlap_name) + np.array(overlap_addr)
            feat['nums_conflict_flag'] = ((s1_na.apply(len) > 0) & (c_na.apply(len) > 0) & (feat['nums_addr_jaccard'] < 1.0)).astype(int)
        else:
            feat['nums_name_jaccard'] = 0.0
            feat['nums_addr_jaccard'] = 0.0
            feat['nums_overlap_count'] = 0
            feat['nums_conflict_flag'] = 0

        if 's1_street_num' in df.columns:
            s1_sn = df['s1_street_num'].fillna('')
            c_sn = df['c_street_num'].fillna('')
            feat['street_num_match'] = ((s1_sn == c_sn) & (s1_sn != "")).astype(int)
            feat['street_num_conflict'] = ((s1_sn != c_sn) & (s1_sn != "") & (c_sn != "")).astype(int)
        if 's1_postal_code' in df.columns:
            s1_pc = df['s1_postal_code'].fillna('')
            c_pc = df['c_postal_code'].fillna('')
            feat['postal_code_match'] = ((s1_pc == c_pc) & (s1_pc != "")).astype(int)
            feat['postal_code_conflict'] = ((s1_pc != c_pc) & (s1_pc != "") & (c_pc != "")).astype(int)
        if 's1_country_norm' in df.columns:
            s1_c = df['s1_country_norm'].fillna('unknown')
            c_c = df['c_country_norm'].fillna('unknown')
            feat['country_match'] = ((s1_c == c_c) & (s1_c != 'unknown')).astype(int)
            feat['country_conflict'] = ((s1_c != c_c) & (s1_c != 'unknown') & (c_c != 'unknown')).astype(int)

        feat['name_len_diff_ratio'] = (n1.str.len() - n2.str.len()).abs() / np.maximum(1, np.maximum(n1.str.len(), n2.str.len()))
        feat['addr_len_diff_ratio'] = (a1.str.len() - a2.str.len()).abs() / np.maximum(1, np.maximum(a1.str.len(), a2.str.len()))

        feat['target_source'] = np.where(df['candidate_entity_id'].astype(str).str.startswith('S2'), 1, 2)

        # --- dense embedding similarity (reused from blocking when available) ---
        embeddings = self._ensure_embeddings(df_s1, df_cand, embeddings)
        s1_ids = df['source1_entity_id'].values
        c_ids = df['candidate_entity_id'].values
        dim = len(next(iter(embeddings.values()))) if embeddings else 0
        if dim:
            s1_mat = np.array([embeddings.get(i, np.zeros(dim)) for i in s1_ids], dtype='float32')
            c_mat = np.array([embeddings.get(i, np.zeros(dim)) for i in c_ids], dtype='float32')
            s1_norm = np.linalg.norm(s1_mat, axis=1)
            c_norm = np.linalg.norm(c_mat, axis=1)
            denom = np.where((s1_norm * c_norm) == 0, 1.0, s1_norm * c_norm)
            cos = np.einsum('ij,ij->i', s1_mat, c_mat) / denom
            feat['combined_emb_cos'] = cos
        else:
            feat['combined_emb_cos'] = 0.0

        # --- retrieval scores from blocking (nearly free signal -- these were
        # computed already during candidate generation and previously discarded),
        # plus the rank/margin/mutual-nearest-neighbor columns from
        # blocking.enrich_pair_scores. All of these MUST come from the full,
        # pre-split, pre-downsampling candidate pool (see enrich_pair_scores'
        # docstring) -- computing them here on whatever subset of pairs this
        # particular call happens to cover would silently differ between a
        # downsampled training call and a full-pool inference call.
        score_cols = ['tfidf_score', 'dense_score', 'retrieval_score', 'retrieval_margin_in_source', 'is_mutual_top1']
        rank_cols = ['retrieval_rank_in_source', 'rank_for_candidate']

        if pair_scores_df is not None and not pair_scores_df.empty:
            available = [c for c in score_cols + rank_cols if c in pair_scores_df.columns]
            feat = feat.merge(pair_scores_df[['source1_entity_id', 'candidate_entity_id'] + available],
                               on=['source1_entity_id', 'candidate_entity_id'], how='left')

        for c in score_cols:
            feat[c] = feat[c].fillna(0.0).astype('float64') if c in feat.columns else 0.0
        for c in rank_cols:
            # 9999 = "never retrieved" -- worse than any real rank, so a pair
            # missing from the enriched pool (e.g. a manually-injected
            # missing-gt-match training row) never looks like a top candidate.
            feat[c] = feat[c].fillna(9999.0).astype('float64') if c in feat.columns else 9999.0

        return feat
