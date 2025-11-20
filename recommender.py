
from typing import Optional, List, Tuple
import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from utils import build_search_text, TEXT_COLS, NUM_COLS

class CarRecommender:
    def __init__(self, max_features: int = 5000):
        self.tfidf = TfidfVectorizer(max_features=max_features, ngram_range=(1,2))
        self.scaler = StandardScaler(with_mean=False)  # with sparse hstack, keep False
        self.nn = NearestNeighbors(metric="cosine", algorithm="brute")
        self.fitted = False
        self._df = None

    def fit(self, df: pd.DataFrame):
        self._df = df.reset_index(drop=True)
        text = build_search_text(self._df)
        X_text = self.tfidf.fit_transform(text)

        X_num = self.scaler.fit_transform(self._df[NUM_COLS].astype(float).values)
        X = hstack([X_text, X_num])
        self.nn.fit(X)
        self.fitted = True
        return self

    def _vectorize_query(self, text_query: str = "", year: Optional[int] = None, kms: Optional[int] = None):
        # Build a one-row vector from text + numeric features
        txt_vec = self.tfidf.transform([text_query])
        num = np.array([[year if year is not None else self._df["year"].median(),
                         kms if kms is not None else self._df["kms_driven"].median()]], dtype=float)
        num_vec = self.scaler.transform(num)
        return hstack([txt_vec, num_vec])

    def recommend_similar_to_index(self, idx: int, k: int = 10) -> List[Tuple[int, float]]:
        assert self.fitted, "Call fit() first."
        # Use the actual precomputed vector of that item by rebuilding its text and numeric
        text = build_search_text(self._df.iloc[[idx]])
        X_text = self.tfidf.transform(text)
        X_num = self.scaler.transform(self._df.iloc[[idx]][NUM_COLS].astype(float).values)
        X = hstack([X_text, X_num])
        dists, inds = self.nn.kneighbors(X, n_neighbors=min(k+1, len(self._df))) # include itself
        inds = inds[0].tolist()
        dists = dists[0].tolist()
        pairs = [(i, d) for i, d in zip(inds, dists) if i != idx]
        return pairs[:k]

    def recommend_by_query(self, text_query: str, year: Optional[int], kms: Optional[int], k:int=10):
        assert self.fitted, "Call fit() first."
        q_vec = self._vectorize_query(text_query, year, kms)
        dists, inds = self.nn.kneighbors(q_vec, n_neighbors=min(k, len(self._df)))
        return list(zip(inds[0].tolist(), dists[0].tolist()))

    @property
    def df(self):
        return self._df.copy()
