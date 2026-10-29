"""
Search ICD-10-CM codes with free text.

- WordRetriever   BM25 over words (description + parent context). Precise on exact terminology.
- CharRetriever   TF-IDF over character 3-5-grams. Robust to morphology and spelling:
                  "myelophthisic" still matches "myelophthisis".
- HybridRetriever Reciprocal Rank Fusion of both.
"""
from __future__ import annotations

import re

import numpy as np
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer

from icd10.codeset import Code

STOP = set("""of the and or in with without to a an by for due other unspecified nos not elsewhere classified
on as at from site part""".split())
TOKEN = re.compile(r"[a-z0-9]+")


def stem(t: str) -> str:
    for suf in ("ies", "es", "s", "ic", "al", "ia", "y"):
        if t.endswith(suf) and len(t) - len(suf) >= 4:
            return t[: -len(suf)]
    return t


def words(text: str, keep_stop: bool = False) -> list[str]:
    return [stem(w) for w in TOKEN.findall(text.lower()) if keep_stop or w not in STOP]


class WordRetriever:
    name = "BM25 (words)"

    def __init__(self, codes: list[Code], with_inclusion_terms: bool = True):
        self.codes = codes
        self.bm25 = BM25Okapi([words(c.search_text(with_inclusion_terms)) for c in codes], k1=1.2, b=0.5)

    def scores(self, query: str) -> np.ndarray:
        return self.bm25.get_scores(words(query))


class CharRetriever:
    name = "TF-IDF (char n-grams)"

    def __init__(self, codes: list[Code], with_inclusion_terms: bool = True):
        self.codes = codes
        self.vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True, min_df=2, lowercase=True)
        self.matrix = self.vec.fit_transform([c.search_text(with_inclusion_terms) for c in codes])

    def scores(self, query: str) -> np.ndarray:
        return (self.matrix @ self.vec.transform([query]).T).toarray().ravel()


def top_k(scores: np.ndarray, k: int) -> np.ndarray:
    idx = np.argpartition(-scores, min(k, len(scores) - 1))[:k]
    return idx[np.argsort(-scores[idx])]


class HybridRetriever:
    name = "Hybrid (RRF)"

    def __init__(self, codes: list[Code], with_inclusion_terms: bool = True, rrf_k: int = 60, depth: int = 100):
        self.codes = codes
        self.parts = [WordRetriever(codes, with_inclusion_terms), CharRetriever(codes, with_inclusion_terms)]
        self.rrf_k, self.depth = rrf_k, depth

    def scores(self, query: str) -> np.ndarray:
        fused = np.zeros(len(self.codes))
        for r in self.parts:
            for rank, i in enumerate(top_k(r.scores(query), self.depth), 1):
                fused[i] += 1.0 / (self.rrf_k + rank)
        return fused


def search(retriever, query: str, k: int = 10) -> list[tuple[Code, float]]:
    s = retriever.scores(query)
    return [(retriever.codes[i], float(s[i])) for i in top_k(s, k)]
