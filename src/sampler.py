"""Phase 4d (part 1): choose which logs of a cluster go into one LLM batch.

Similarity is the cosine of TF-IDF vectors built from the *unmasked* tokens of the cluster's
logs, so logs that differ only in their variables still look different. That is the variability
the batch is meant to show the LLM.
"""
import random

import numpy as np
import regex as re
from sklearn.metrics.pairwise import cosine_similarity

from src.partition import vectorize

_RAW_DELIMS = re.compile(r"([=:,()\[\]{}\"'|;\s])")


def _raw_tokens(content):
    return [t for t in _RAW_DELIMS.split(content) if t and not t.isspace()]


def similarity_matrix(logs):
    X = vectorize([_raw_tokens(l) or ["<empty>"] for l in logs])
    return cosine_similarity(X)


def dpp_greedy(kernel, k, eps=1e-10):
    """Greedy MAP inference for a DPP (Chen, Zhang & Zhou, NeurIPS 2018): repeatedly add the item
    that most increases log det(L_S). Returns selected indices in selection order."""
    n = kernel.shape[0]
    k = min(k, n)
    cis = np.zeros((k, n))
    di2s = np.array(np.diag(kernel), dtype=float)
    selected = [int(np.argmax(di2s))]
    while len(selected) < k:
        t = len(selected) - 1
        j = selected[-1]
        eis = (kernel[j, :] - cis[:t, j] @ cis[:t, :]) / np.sqrt(di2s[j])
        cis[t, :] = eis
        di2s = di2s - eis ** 2
        di2s[selected] = -np.inf
        nxt = int(np.argmax(di2s))
        if di2s[nxt] < eps:
            break
        selected.append(nxt)
    return selected


def _farthest(sim, k):
    selected = [0]
    while len(selected) < k:
        closest = sim[:, selected].max(axis=1)
        closest[selected] = np.inf
        selected.append(int(np.argmin(closest)))
    return selected


def _most_similar(sim, k):
    """Similarity sampling: the most central log plus its nearest neighbours."""
    centre = int(np.argmax(sim.sum(axis=1)))
    order = np.argsort(-sim[centre])
    return [int(i) for i in order[:k]]


def sample(logs, k, method="dpp", seed=0):
    """Pick up to k logs from `logs` (unique contents). Clusters smaller than k are used whole."""
    if len(logs) <= k:
        return list(logs)
    if k == 1:
        return [logs[0]]
    if method == "random":
        rng = random.Random(seed + len(logs))
        return rng.sample(list(logs), k)
    sim = similarity_matrix(logs)
    if method == "dpp":
        idx = dpp_greedy(sim + 1e-6 * np.eye(len(logs)), k)
        if len(idx) < k:   # kernel rank exhausted: top up with farthest-point picks
            rest = [i for i in _farthest(sim, len(logs)) if i not in idx]
            idx += rest[: k - len(idx)]
    elif method == "farthest":
        idx = _farthest(sim, k)
    elif method == "similarity":
        idx = _most_similar(sim, k)
    else:
        raise ValueError(f"unknown sampler {method!r}")
    return [logs[i] for i in idx]
