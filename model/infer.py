"""Self-contained inference for poker44-aquila (D0Aquila).

Loads the trained artifact, extracts phasberg + v2 features (identical code
paths to training), weight-blends the four members by rank, then applies a
rank-preserving threshold remap plus a batch safety budget. Ranking (and
therefore AP / recall@FPR) is never altered by the post-processing — only the
0.5 decision boundary moves.
"""
from __future__ import annotations

import hashlib
import json
import os
import pickle
from typing import Any, Dict, List

import numpy as np

from d0_features import phasberg_dict, v2_dict
from d0_ensemble import D0Aquila  # noqa: F401  (needed to unpickle the artifact)
from d0_drse import DRSE  # noqa: F401  (DRSE instances live inside the pickle)

_ART_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "artifacts")
_ARTIFACT = os.environ.get("POKER44_ARTIFACT", "d0_aquila_v39.pkl")
_MAX_POS_FRAC = float(os.environ.get("POKER44_MAX_POS_FRAC", "0.145"))


def artifact_path() -> str:
    return os.path.join(_ART_DIR, _ARTIFACT)


def artifact_sha256() -> str:
    digest = hashlib.sha256()
    with open(artifact_path(), "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _apply_batch_safety_budget(scores: np.ndarray, max_frac: float) -> np.ndarray:
    """Cap the fraction of >=0.5 calls per batch WITHOUT changing the ranking.

    Only scores already past the decision boundary can be compressed, and they
    are compressed into the open interval between the highest sub-threshold
    score and 0.5 — sub-threshold calibration is untouched, the global
    ordering is preserved exactly, and at least one confident call always
    survives (k >= 1) even on tiny batches.
    """
    s = np.asarray(scores, dtype=float)
    n = s.size
    if n == 0 or max_frac >= 1.0:
        return s
    k = max(1, int(np.floor(max_frac * n)))
    positive = np.flatnonzero(s >= 0.5)
    if positive.size <= k:
        return s
    order = positive[np.argsort(-s[positive], kind="stable")]
    squeeze = order[k:]
    below = s[s < 0.5]
    lo = min(float(below.max()) if below.size else 0.45, 0.499)
    span = 0.5 - lo
    out = s.copy()
    m = squeeze.size
    for rank, idx in enumerate(squeeze):
        out[idx] = lo + span * (m - rank) / (m + 1.0)
    return np.clip(out, 0.0, 1.0)


def _remap_to_threshold(p: np.ndarray, t: float) -> np.ndarray:
    """Monotone piecewise-linear remap moving decision threshold t to 0.5."""
    t = float(min(max(t, 1e-6), 1 - 1e-6))
    out = np.where(p >= t, 0.5 + 0.5 * (p - t) / (1 - t), 0.5 * p / t)
    return np.clip(out, 0.0, 1.0)


class Poker44Model:
    def __init__(self, art_dir: str = _ART_DIR):
        with open(os.path.join(art_dir, _ARTIFACT), "rb") as fh:
            self.ens = pickle.load(fh)
        with open(os.path.join(art_dir, "meta.json")) as fh:
            self.meta = json.load(fh)
        self.threshold: float = float(self.meta["deploy_threshold"])
        self.artifact_name: str = _ARTIFACT
        self.cph = self.ens.cols_ph
        self.cv2 = self.ens.cols_v2

    def _matrices(self, chunks):
        ph = np.array([[float(d.get(c, 0.0)) for c in self.cph]
                       for d in (phasberg_dict(c) for c in chunks)], dtype=float)
        v2 = np.array([[float(d.get(c, 0.0)) for c in self.cv2]
                       for d in (v2_dict(c) for c in chunks)], dtype=float)
        return ph, v2

    def score_chunks(self, chunks: List[List[Dict[str, Any]]]) -> List[float]:
        if not chunks:
            return []
        ph, v2 = self._matrices(chunks)
        p = self.ens.score(ph, v2)
        scores = _remap_to_threshold(np.asarray(p, dtype=float), self.threshold)
        scores = _apply_batch_safety_budget(scores, _MAX_POS_FRAC)
        return [0.1 if not chunk else round(float(s), 6)
                for chunk, s in zip(chunks, scores)]


_SINGLETON: Poker44Model | None = None


def get_model() -> Poker44Model:
    global _SINGLETON
    if _SINGLETON is None:
        _SINGLETON = Poker44Model()
    return _SINGLETON
