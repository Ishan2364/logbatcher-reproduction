"""Phase 4b: partition logs with TF-IDF + DBSCAN (cosine), outliers last.

partition() returns a list of clusters; each cluster is a list of *unique* contents (duplicates
removed), largest cluster first, DBSCAN outliers as one final cluster.
"""
from collections import Counter

import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.feature_extraction.text import TfidfVectorizer

from src.preprocess import clustering_tokens


def _identity(x):
    return x


def vectorize(token_lists):
    vec = TfidfVectorizer(tokenizer=_identity, preprocessor=_identity, lowercase=False, token_pattern=None)
    return vec.fit_transform(token_lists)


def _dedup(contents):
    seen, out = set(), []
    for c in contents:
        if c not in seen:
            seen.add(c)
            out.append(c)
    return out


def partition(contents, eps=0.5, min_samples=5, metric="cosine"):
    """contents: list of log messages (one per line, duplicates allowed).

    Clustering runs on all lines so duplicates count towards DBSCAN density, as in the paper;
    each returned cluster is de-duplicated. Size ordering uses the number of lines.
    """
    X = vectorize([clustering_tokens(c) for c in contents])
    labels = DBSCAN(eps=eps, min_samples=min_samples, metric=metric).fit_predict(X)
    groups = {}
    for c, lab in zip(contents, labels):
        groups.setdefault(int(lab), []).append(c)
    outliers = groups.pop(-1, [])
    ordered = sorted(groups.values(), key=len, reverse=True)
    clusters = [_dedup(g) for g in ordered]
    if outliers:
        clusters.append(_dedup(outliers))
    return clusters, labels


def chunk_partition(contents, chunk_size):
    """Ablation 'w/o partitioning': fixed-size consecutive windows instead of clustering."""
    return [_dedup(contents[i:i + chunk_size]) for i in range(0, len(contents), chunk_size)]


def cluster_summary(clusters, contents):
    counts = Counter(contents)
    sizes = [sum(counts[c] for c in cl) for cl in clusters]
    return {"n_clusters": len(clusters), "largest": max(sizes) if sizes else 0,
            "mean_unique": float(np.mean([len(c) for c in clusters])) if clusters else 0.0}
