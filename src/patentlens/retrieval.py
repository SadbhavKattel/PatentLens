"""Retrieval models for patent similarity search.

All retrievers share the same small interface so the training notebook and the
Streamlit app can treat them interchangeably:

    retriever.fit(texts)                      -> self
    retriever.rank(query_idx, top_k)           -> (doc_indices, scores), query excluded
    retriever.rank_text(free_text, top_k)      -> (doc_indices, scores)   # for live search
    retriever.save(path) / Class.load(path)

Similarity is always computed for ONE query against the corpus (a single row/vector),
never as a full n x n matrix. That's what lets this scale from 3,000 to 50,000+ patents
without blowing up memory or runtime -- the original notebook's biggest scaling problem.
"""

from pathlib import Path

import joblib
import numpy as np
import scipy.sparse as sp
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def _exclude_self(order, query_idx):
    return order[order != query_idx]


class TfidfRetriever:
    def __init__(self, max_features=5000, ngram_range=(1, 2), min_df=2, name="TF-IDF"):
        self.vectorizer = TfidfVectorizer(
            max_features=max_features, ngram_range=ngram_range, min_df=min_df
        )
        self.matrix = None
        self.name = name

    def fit(self, texts):
        self.matrix = self.vectorizer.fit_transform(list(texts))
        return self

    def rank(self, query_idx, top_k=10, exclude_self=True):
        sims = cosine_similarity(self.matrix[query_idx], self.matrix).ravel()
        order = np.argsort(-sims)
        if exclude_self:
            order = _exclude_self(order, query_idx)
        order = order[:top_k]
        return order.tolist(), sims[order].tolist()

    def rank_text(self, text, top_k=10):
        vec = self.vectorizer.transform([text])
        sims = cosine_similarity(vec, self.matrix).ravel()
        order = np.argsort(-sims)[:top_k]
        return order.tolist(), sims[order].tolist()

    def save(self, path):
        joblib.dump({'vectorizer': self.vectorizer, 'matrix': self.matrix, 'name': self.name}, path)

    @classmethod
    def load(cls, path):
        data = joblib.load(path)
        obj = cls(name=data['name'])
        obj.vectorizer = data['vectorizer']
        obj.matrix = data['matrix']
        return obj


class LsaRetriever:
    def __init__(self, tfidf: TfidfRetriever, n_components=100, name="LSA"):
        self.tfidf = tfidf
        self.svd = TruncatedSVD(n_components=n_components, random_state=42)
        self.lsa_matrix = None
        self.name = name

    def fit(self, texts=None):
        # Reuses the already-fit TF-IDF matrix; texts is accepted for interface symmetry.
        self.lsa_matrix = self.svd.fit_transform(self.tfidf.matrix)
        return self

    def rank(self, query_idx, top_k=10, exclude_self=True):
        sims = cosine_similarity(
            self.lsa_matrix[query_idx:query_idx + 1], self.lsa_matrix
        ).ravel()
        order = np.argsort(-sims)
        if exclude_self:
            order = _exclude_self(order, query_idx)
        order = order[:top_k]
        return order.tolist(), sims[order].tolist()

    def rank_text(self, text, top_k=10):
        vec = self.tfidf.vectorizer.transform([text])
        lsa_vec = self.svd.transform(vec)
        sims = cosine_similarity(lsa_vec, self.lsa_matrix).ravel()
        order = np.argsort(-sims)[:top_k]
        return order.tolist(), sims[order].tolist()

    def save(self, path):
        joblib.dump({'svd': self.svd, 'lsa_matrix': self.lsa_matrix, 'name': self.name}, path)

    @classmethod
    def load(cls, path, tfidf: TfidfRetriever):
        data = joblib.load(path)
        obj = cls(tfidf, name=data['name'])
        obj.svd = data['svd']
        obj.lsa_matrix = data['lsa_matrix']
        return obj


class Bm25Retriever:
    """BM25 via a precomputed sparse matrix, not the rank_bm25 library.

    rank_bm25.BM25Okapi.get_scores() loops over every document with Python-level dict
    lookups per query term -- O(corpus) Python overhead per query, which is fine at a few
    hundred queries but becomes a genuine bottleneck at thousands (measured: ~0.35s/query
    at 100k documents, so a 10k-query evaluation alone would take nearly an hour).

    This does the same BM25 math (Robertson-Sparck Jones idf, standard k1/b saturation)
    but precomputes each document's saturated, idf-weighted term scores into one sparse
    matrix at fit time. Ranking a query then becomes a single sparse matrix-vector
    multiply -- the same trick that makes TF-IDF fast -- instead of a Python loop.
    """

    def __init__(self, k1=1.5, b=0.75, name="BM25"):
        self.k1 = k1
        self.b = b
        self.name = name
        self.vectorizer = None
        self.term_counts = None
        self.bm25_matrix = None

    def fit(self, texts):
        from sklearn.feature_extraction.text import CountVectorizer

        self.vectorizer = CountVectorizer()
        tf = self.vectorizer.fit_transform(list(texts)).tocsr()
        self.term_counts = tf

        n_docs = tf.shape[0]
        doc_len = np.asarray(tf.sum(axis=1)).ravel()
        avgdl = doc_len.mean()

        doc_freq = np.diff(tf.tocsc().indptr)
        idf = np.log(1 + (n_docs - doc_freq + 0.5) / (doc_freq + 0.5))

        coo = tf.tocoo()
        data = coo.data.astype(np.float64)
        doc_len_per_nnz = doc_len[coo.row]
        denom = data + self.k1 * (1 - self.b + self.b * doc_len_per_nnz / avgdl)
        saturated = data * (self.k1 + 1) / denom
        weighted = saturated * idf[coo.col]

        self.bm25_matrix = sp.csr_matrix((weighted, (coo.row, coo.col)), shape=tf.shape)
        return self

    def _score_from_vector(self, presence_vec):
        scores = self.bm25_matrix @ presence_vec.T
        return np.asarray(scores.todense()).ravel()

    def _binary(self, sparse_vec):
        v = sparse_vec.copy()
        v.data[:] = 1.0
        return v

    def rank(self, query_idx, top_k=10, exclude_self=True):
        scores = self._score_from_vector(self._binary(self.term_counts[query_idx]))
        order = np.argsort(-scores)
        if exclude_self:
            order = _exclude_self(order, query_idx)
        order = order[:top_k]
        return order.tolist(), scores[order].tolist()

    def rank_text(self, text, top_k=10):
        query_vec = self.vectorizer.transform([text])
        scores = self._score_from_vector(self._binary(query_vec))
        order = np.argsort(-scores)[:top_k]
        return order.tolist(), scores[order].tolist()

    def save(self, path):
        joblib.dump({
            'vectorizer': self.vectorizer,
            'term_counts': self.term_counts,
            'bm25_matrix': self.bm25_matrix,
            'k1': self.k1,
            'b': self.b,
            'name': self.name,
        }, path)

    @classmethod
    def load(cls, path):
        data = joblib.load(path)
        obj = cls(k1=data['k1'], b=data['b'], name=data['name'])
        obj.vectorizer = data['vectorizer']
        obj.term_counts = data['term_counts']
        obj.bm25_matrix = data['bm25_matrix']
        return obj


class EmbeddingRetriever:
    """Sentence-embedding retriever backed by a FAISS flat inner-product index
    (cosine similarity, since embeddings are L2-normalized). Scales to large corpora
    far better than a dense n x n similarity matrix.
    """

    def __init__(self, model_name="all-MiniLM-L6-v2", name=None):
        self.model_name = model_name
        self.name = name or model_name
        self.model = None
        self.embeddings = None
        self.index = None

    def _get_model(self):
        if self.model is None:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(self.model_name)
        return self.model

    def fit(self, texts, batch_size=32, show_progress_bar=True):
        import faiss

        model = self._get_model()
        emb = model.encode(
            list(texts),
            batch_size=batch_size,
            show_progress_bar=show_progress_bar,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        self.embeddings = emb.astype('float32')
        self.index = faiss.IndexFlatIP(self.embeddings.shape[1])
        self.index.add(self.embeddings)
        return self

    def rank(self, query_idx, top_k=10, exclude_self=True):
        k = min(top_k + (1 if exclude_self else 0), len(self.embeddings))
        scores, idxs = self.index.search(self.embeddings[query_idx:query_idx + 1], k)
        idxs, scores = idxs[0], scores[0]
        if exclude_self:
            mask = idxs != query_idx
            idxs, scores = idxs[mask], scores[mask]
        idxs, scores = idxs[:top_k], scores[:top_k]
        return idxs.tolist(), scores.tolist()

    def rank_text(self, text, top_k=10):
        model = self._get_model()
        vec = model.encode([text], convert_to_numpy=True, normalize_embeddings=True).astype('float32')
        scores, idxs = self.index.search(vec, top_k)
        return idxs[0].tolist(), scores[0].tolist()

    def save(self, dirpath):
        import faiss

        dirpath = Path(dirpath)
        dirpath.mkdir(parents=True, exist_ok=True)
        np.save(dirpath / 'embeddings.npy', self.embeddings)
        faiss.write_index(self.index, str(dirpath / 'index.faiss'))
        joblib.dump({'model_name': self.model_name, 'name': self.name}, dirpath / 'meta.joblib')

    @classmethod
    def load(cls, dirpath, load_model=True):
        import faiss

        dirpath = Path(dirpath)
        meta = joblib.load(dirpath / 'meta.joblib')
        obj = cls(model_name=meta['model_name'], name=meta['name'])
        obj.embeddings = np.load(dirpath / 'embeddings.npy')
        obj.index = faiss.read_index(str(dirpath / 'index.faiss'))
        if load_model:
            obj._get_model()
        return obj


class HybridRetriever:
    """Combines multiple retrievers via Reciprocal Rank Fusion (RRF).

    RRF just adds 1/(rrf_k + rank) across each retriever's ranked list per document --
    no score normalization needed, which matters here since TF-IDF/BM25/cosine scores
    aren't on comparable scales. rrf_k=60 is the standard default from the original
    RRF paper (Cormack et al.) and is not sensitive to tuning.
    """

    def __init__(self, retrievers, weights=None, rrf_k=60, name="Hybrid (RRF)", pool_size=100):
        self.retrievers = retrievers
        self.weights = weights or [1.0] * len(retrievers)
        self.rrf_k = rrf_k
        self.name = name
        self.pool_size = pool_size

    def _fuse(self, per_retriever_rankings, top_k):
        fused = {}
        for ranked_idxs, weight in zip(per_retriever_rankings, self.weights):
            for pos, idx in enumerate(ranked_idxs):
                fused[idx] = fused.get(idx, 0.0) + weight / (self.rrf_k + pos + 1)
        ordered = sorted(fused.items(), key=lambda kv: -kv[1])[:top_k]
        return [i for i, _ in ordered], [s for _, s in ordered]

    def rank(self, query_idx, top_k=10, exclude_self=True):
        rankings = [
            r.rank(query_idx, top_k=self.pool_size, exclude_self=exclude_self)[0]
            for r in self.retrievers
        ]
        return self._fuse(rankings, top_k)

    def rank_text(self, text, top_k=10):
        rankings = [r.rank_text(text, top_k=self.pool_size)[0] for r in self.retrievers]
        return self._fuse(rankings, top_k)
