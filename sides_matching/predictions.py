from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
from scipy.spatial.distance import cdist
from sklearn.metrics.pairwise import cosine_similarity
from wildlife_tools.similarity import MatchLightGlue
from wildlife_tools.similarity.calibration import IsotonicCalibration

from .utils import get_features, unique_no_sort


def as_feature_matrix(features) -> np.ndarray:
    """Accept wildlife FeatureDataset pickles or raw ndarray feature matrices."""
    if hasattr(features, "features"):
        return np.asarray(features.features)
    return np.asarray(features)


def _apply_ignore(similarity: np.ndarray, ignore=None) -> np.ndarray:
    """Copy similarity and set ignored entries to -inf."""
    out = np.array(similarity, copy=True, dtype=np.float64)
    if ignore is None:
        return out
    if ignore == "diagonal":
        if out.shape[0] != out.shape[1]:
            raise Exception("For ignore=diagonal, query and database must correspond")
        ignore = [[i] for i in range(out.shape[0])]
    for i in range(len(ignore)):
        out[i, ignore[i]] = -np.inf
    return out


def _finite_pair_labels(
    similarity: np.ndarray,
    identity: np.ndarray,
    query_indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Flatten finite (query, database) pairs into scores and same-identity hits."""
    scores: list[np.ndarray] = []
    hits: list[np.ndarray] = []
    for i in query_indices:
        row = similarity[i]
        finite = np.isfinite(row)
        if not finite.any():
            continue
        js = np.flatnonzero(finite)
        scores.append(row[js].astype(np.float64, copy=False))
        hits.append((identity[i] == identity[js]).astype(np.float64))
    if not scores:
        raise ValueError("No finite pairs available for calibration fit")
    return np.concatenate(scores), np.concatenate(hits)


def _calibrate_matrix(calibrator: IsotonicCalibration, similarity: np.ndarray) -> np.ndarray:
    """Apply a fitted calibrator to finite entries; keep non-finite as -inf."""
    out = np.full(similarity.shape, -np.inf, dtype=np.float64)
    finite = np.isfinite(similarity)
    if finite.any():
        out[finite] = calibrator.predict(similarity[finite])
    return out


def _split_query_indices(
    n: int,
    val_fraction: float,
    seed: int,
    val_indices: Optional[Sequence[int]] = None,
) -> tuple[np.ndarray, np.ndarray]:
    if val_indices is not None:
        val = np.unique(np.asarray(val_indices, dtype=np.int64))
        if val.size == 0 or val.min() < 0 or val.max() >= n:
            raise ValueError("val_indices must be non-empty indices into the query axis")
        mask = np.ones(n, dtype=bool)
        mask[val] = False
        test = np.flatnonzero(mask)
        if test.size == 0:
            raise ValueError("val_indices left no held-out test queries")
        return val, test

    if not 0.0 < val_fraction < 1.0:
        raise ValueError("val_fraction must be in (0, 1)")
    if n < 2:
        raise ValueError("Need at least 2 queries to hold out a calibration split")

    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n_val = int(round(n * val_fraction))
    n_val = min(max(n_val, 1), n - 1)
    return np.sort(perm[:n_val]), np.sort(perm[n_val:])


class Data():
    def __init__(self, path_features_query, path_features_database):
        self.path_features_query = path_features_query
        self.path_features_database = path_features_database

    def get_features(self):
        return get_features(self.path_features_query), get_features(self.path_features_database)

    def compute_similarity(self, ignore=None):
        features_query, features_database = self.get_features()
        # Compute the cosine similarity between the query and the database
        similarity = self.matcher(features_query, features_database)
        # Set -infinity for ignored indices
        if ignore is not None:
            if ignore == 'diagonal':
                n_query = len(as_feature_matrix(features_query))
                n_database = len(as_feature_matrix(features_database))
                if n_query == n_database:
                    ignore = [[i] for i in range(n_query)]
                else:
                    raise Exception('For ignore=diagonal, query and database must correspond')
            for i in range(len(ignore)):
                similarity[i, ignore[i]] = -np.inf
        return similarity

class MegaDescriptor(Data):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Prefer array-based cosine so LoRA ndarray pickles and DeepFeatures
        # FeatureDataset pickles both work.
        self.matcher = self._cosine_similarity

    @staticmethod
    def _cosine_similarity(query, database):
        return cosine_similarity(as_feature_matrix(query), as_feature_matrix(database))

class Aliked(Data):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.matcher = MatchLightGlue('aliked')

class Sift(Data):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.matcher = MatchLightGlue('sift')

class TORSOOI(Data):
    def __init__(self, df):
        self.df = df
        self.matcher = lambda x, y: cdist(x, y, lambda a, b: sum(a==b))

    def get_features(self):
        features = np.array([list(x) for x in self.df['id_code'].to_numpy()])
        return features, features

class Combined:
    """Fuse two similarity matrices via max or WildFusion-style calibration.

    ``method="calibrated"`` fits isotonic+PCHIP calibrators on a held-out
    subset of query rows, then returns the equal-weight average of calibrated
    scores. ``method="max"`` keeps the legacy uncalibrated maximum.
    """

    def __init__(
        self,
        prediction0,
        prediction1,
        *,
        val_fraction: float = 0.5,
        seed: int = 0,
        method: str = "calibrated",
        val_indices: Optional[Sequence[int]] = None,
    ):
        if not prediction0.df.equals(prediction1.df):
            raise Exception("Dataframes are not equal")
        if prediction0.similarity.shape != prediction1.similarity.shape:
            raise Exception("Similarity does not have the same size")
        if method not in {"calibrated", "max"}:
            raise ValueError(f"Unknown method: {method!r}")

        self.prediction0 = prediction0
        self.prediction1 = prediction1
        self.method = method
        self.val_fraction = val_fraction
        self.seed = seed
        self.calibrators: Optional[list[IsotonicCalibration]] = None

        n_query = prediction0.similarity.shape[0]
        if method == "max":
            self.val_indices = np.array([], dtype=np.int64)
            self.test_indices = np.arange(n_query, dtype=np.int64)
        else:
            self.val_indices, self.test_indices = _split_query_indices(
                n_query, val_fraction, seed, val_indices=val_indices
            )
            self._fit_calibrators()

    def _fit_calibrators(self) -> None:
        identity = self.prediction0.identity
        calibrators: list[IsotonicCalibration] = []
        for prediction in (self.prediction0, self.prediction1):
            scores, hits = _finite_pair_labels(
                prediction.similarity, identity, self.val_indices
            )
            if hits.min() == hits.max():
                raise ValueError(
                    "Calibration fit requires both positive and negative pairs "
                    "in the validation query split"
                )
            calibrator = IsotonicCalibration()
            calibrator.fit(scores, hits)
            calibrators.append(calibrator)
        self.calibrators = calibrators

    def compute_similarity(self, ignore=None, **kwargs):
        del kwargs  # accept Data-style kwargs for call-site compatibility
        if self.method == "max":
            fused = np.maximum(
                self.prediction0.similarity, self.prediction1.similarity
            )
            return _apply_ignore(fused, ignore)

        assert self.calibrators is not None
        s0 = _calibrate_matrix(self.calibrators[0], self.prediction0.similarity)
        s1 = _calibrate_matrix(self.calibrators[1], self.prediction1.similarity)
        fused = np.full(s0.shape, -np.inf, dtype=np.float64)
        both = np.isfinite(s0) & np.isfinite(s1)
        fused[both] = 0.5 * (s0[both] + s1[both])
        return _apply_ignore(fused, ignore)


class Prediction():
    def __init__(self, df, similarity, query_indices=None, **kwargs):
        self.df = df
        self.identity = df['identity'].to_numpy()
        self.orientation = df['orientation'].to_numpy()
        self.year = df['year'].to_numpy()
        self.n_individuals = df['identity'].nunique()
        self.similarity = similarity
        if query_indices is None:
            self.query_indices = np.arange(len(similarity), dtype=np.int64)
        else:
            self.query_indices = np.asarray(query_indices, dtype=np.int64)
        self.compute_scores(**kwargs)
        self.true_label = self.identity[self.true]
        self.pred_label = self.identity[self.pred]

    def compute_scores(self, k=None):
        if k is None:
            k = self.similarity.shape[1]
        self.true = self.query_indices.copy()
        ranked = (-self.similarity[self.true]).argsort(axis=-1)[:, :k]
        self.pred = ranked
        self.scores = np.take_along_axis(self.similarity[self.true], self.pred, axis=-1)

    def compute_accuracy(self, mods):
        metrics = [f'top {i}' for i in range(1, 1+self.n_individuals)]
        accuracy = {mod: {metric: 0 for metric in metrics} for mod in mods}
        
        # Loop over individual query images
        for i, (i_pred, i_true) in enumerate(zip(self.pred, self.true)):
            # Extract identity, orientation and year        
            identity_pred_full = self.identity[i_pred]
            orientation_pred = self.orientation[i_pred]
            year_pred = self.year[i_pred]
            identity_true = self.identity[i_true]
            orientation_true = self.orientation[i_true]
            year_true = self.year[i_true]
            same_identity = identity_pred_full == identity_true
            # Save metrics for individual mods
            for mod in mods:            
                # Select indices to ignore for individual mods
                if mod == 'full':
                    ignore = np.zeros(len(identity_pred_full), dtype='bool')
                elif mod == 'different orientation':
                    ignore = orientation_true == orientation_pred
                elif mod == 'same orientation':
                    ignore = orientation_true != orientation_pred
                elif mod == 'different year':
                    ignore = year_true == year_pred
                elif mod == 'same year':
                    ignore = year_true != year_pred
                elif mod == 'different both':
                    ignore = (orientation_true == orientation_pred) + (year_true == year_pred)
                else:
                    raise Exception('Unknown mod')            
                # Ignore selected indices but only of the same individual
                identity_pred = identity_pred_full[~(same_identity * ignore)]
                # Get the unique predictions
                identity_pred_unique = unique_no_sort(identity_pred)            
                # Compute the metrics
                for i in range(1, 1+self.n_individuals):
                    accuracy[mod][f'top {i}'] += (identity_true in identity_pred_unique[:i]) / len(self.true)
        self.accuracy = accuracy

    def split_scores(self, save_idx=False):
        scores_split = {x: {y: {z: [] for z in {True, False}} for y in {True, False}} for x in {True, False}}
        for i_score, (i_pred, i) in enumerate(zip(self.pred, self.true)):
            for j_score, j in enumerate(i_pred):
                equal_identity = self.identity[i] == self.identity[j]
                equal_orientation = self.orientation[i] == self.orientation[j]
                equal_year = self.year[i] == self.year[j]
                score = self.scores[i_score, j_score]
                if save_idx:
                    score_add = (score, i, j)
                else:
                    score_add = score
                scores_split[equal_identity][equal_orientation][equal_year].append(score_add)
        return scores_split