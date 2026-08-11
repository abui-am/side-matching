"""DeDoDe local feature matching for closed-set re-ID similarity matrices."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from DeDoDe import dedode_descriptor_B, dedode_detector_L
from DeDoDe.matchers.dual_softmax_matcher import DualSoftMaxMatcher
from PIL import Image, ImageOps
from tqdm import tqdm

from sides_matching.loma_matching import resolve_image_paths


@dataclass(frozen=True)
class DeDoDeImageFeatures:
    keypoints: torch.Tensor
    confidence: torch.Tensor
    descriptions: torch.Tensor

    def to_cpu_dict(self) -> dict[str, object]:
        return {
            "keypoints": self.keypoints.detach().cpu(),
            "confidence": self.confidence.detach().cpu(),
            "descriptions": self.descriptions.detach().cpu(),
        }

    @classmethod
    def from_cpu_dict(cls, payload: dict[str, object]) -> DeDoDeImageFeatures:
        return cls(
            keypoints=payload["keypoints"],
            confidence=payload["confidence"],
            descriptions=payload["descriptions"],
        )


def load_dedode_image(
    path: Path,
    *,
    flip: bool,
    size: int,
    normalizer,
    device: torch.device,
) -> torch.Tensor:
    """Load RGB image as DeDoDe tensor (1, 3, H, W) with ImageNet normalization."""
    pil_im = Image.open(path).convert("RGB")
    if flip:
        pil_im = ImageOps.mirror(pil_im)
    pil_im = pil_im.resize((size, size))
    arr = np.asarray(pil_im, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1).float()
    return normalizer(tensor).to(device)[None]


class DeDoDeMatcher:
    """DeDoDe L detector + B descriptor with DualSoftMax match-count similarity."""

    def __init__(
        self,
        *,
        device: str | torch.device,
        num_keypoints: int = 10_000,
        image_size: int = 784,
        inv_temp: float = 20.0,
        threshold: float = 0.1,
    ) -> None:
        self.device = torch.device(device)
        self.num_keypoints = num_keypoints
        self.image_size = image_size
        self.inv_temp = inv_temp
        self.threshold = threshold

        self.detector = dedode_detector_L(device=self.device)
        self.descriptor = dedode_descriptor_B(device=self.device)
        self.matcher = DualSoftMaxMatcher()

    def _image_batch(self, path: Path, *, flip: bool) -> dict[str, torch.Tensor]:
        image = load_dedode_image(
            path,
            flip=flip,
            size=self.image_size,
            normalizer=self.detector.normalizer,
            device=self.device,
        )
        return {"image": image}

    def extract_features(self, path: Path, *, flip: bool = False) -> DeDoDeImageFeatures:
        batch = self._image_batch(path, flip=flip)
        detections = self.detector.detect(batch, num_keypoints=self.num_keypoints)
        descriptions = self.descriptor.describe_keypoints(batch, detections["keypoints"])[
            "descriptions"
        ]
        return DeDoDeImageFeatures(
            detections["keypoints"],
            detections["confidence"],
            descriptions,
        )

    def match_count(self, left: DeDoDeImageFeatures, right: DeDoDeImageFeatures) -> int:
        left_kp = left.keypoints.to(self.device)
        left_desc = left.descriptions.to(self.device)
        left_conf = left.confidence.to(self.device)
        right_kp = right.keypoints.to(self.device)
        right_desc = right.descriptions.to(self.device)
        right_conf = right.confidence.to(self.device)
        matches_a, _, _ = self.matcher.match(
            left_kp,
            left_desc,
            right_kp,
            right_desc,
            P_A=left_conf,
            P_B=right_conf,
            normalize=True,
            inv_temp=self.inv_temp,
            threshold=self.threshold,
        )
        return int(len(matches_a))

    def compute_similarity(
        self,
        query_features: Sequence[DeDoDeImageFeatures],
        database_features: Sequence[DeDoDeImageFeatures],
        *,
        ignore_diagonal: bool = True,
        show_progress: bool = True,
    ) -> np.ndarray:
        n_query = len(query_features)
        n_database = len(database_features)
        sim = np.zeros((n_query, n_database), dtype=np.float64)
        iterator = tqdm(range(n_query), desc="DeDoDe pairs", disable=not show_progress)
        for i in iterator:
            for j in range(n_database):
                if ignore_diagonal and n_query == n_database and i == j:
                    sim[i, j] = -np.inf
                    continue
                sim[i, j] = float(self.match_count(query_features[i], database_features[j]))
        return sim

    def compute_shortlist_similarity(
        self,
        query_features: Sequence[DeDoDeImageFeatures],
        database_features: Sequence[DeDoDeImageFeatures],
        shortlists: Sequence[Sequence[int]],
        *,
        show_progress: bool = True,
    ) -> np.ndarray:
        """Match each query only against MD shortlist indices (O(n·N), scalable)."""
        n_query = len(query_features)
        n_database = len(database_features)
        if len(shortlists) != n_query:
            raise ValueError("shortlists length must match number of queries")
        sim = np.full((n_query, n_database), -np.inf, dtype=np.float64)
        iterator = tqdm(range(n_query), desc="DeDoDe shortlist", disable=not show_progress)
        for i in iterator:
            for j in shortlists[i]:
                sim[i, int(j)] = float(
                    self.match_count(query_features[i], database_features[int(j)])
                )
        return sim


def _feature_cache_path(cache_dir: Path, name: str, flip: bool) -> Path:
    return cache_dir / f"{name}_dedodeLB_flip{flip}_feats.pt"


def _similarity_cache_path(cache_dir: Path, name: str, flip: bool) -> Path:
    return cache_dir / f"{name}_dedodeLB_flip{flip}.npy"


def load_or_build_feature_cache(
    *,
    name: str,
    paths: Sequence[Path],
    flip: bool,
    cache_dir: Path,
    matcher: DeDoDeMatcher,
) -> list[DeDoDeImageFeatures]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache = _feature_cache_path(cache_dir, name, flip)
    if cache.is_file():
        payloads = torch.load(cache, weights_only=False)
        return [DeDoDeImageFeatures.from_cpu_dict(item) for item in payloads]

    features: list[DeDoDeImageFeatures] = []
    for path in tqdm(paths, desc=f"DeDoDe features flip={flip}"):
        features.append(matcher.extract_features(path, flip=flip))
    torch.save([feat.to_cpu_dict() for feat in features], cache)
    return features


def _shortlist_similarity_cache_path(cache_dir: Path, name: str, flip: bool, top_n: int) -> Path:
    return cache_dir / f"{name}_dedodeLB_flip{flip}_N{top_n}.npy"


def md_shortlists(md_sim: np.ndarray, top_n: int) -> list[np.ndarray]:
    """Per-query MegaDescriptor top-N gallery indices (finite scores only)."""
    shortlists: list[np.ndarray] = []
    for i in range(md_sim.shape[0]):
        finite = np.flatnonzero(np.isfinite(md_sim[i]))
        order = finite[np.argsort(-md_sim[i, finite])]
        shortlists.append(order[:top_n])
    return shortlists


def load_or_compute_dedode_shortlist_similarity(
    *,
    name: str,
    data_root: Path,
    paths: Sequence[str],
    flip_query: bool,
    md_sim: np.ndarray,
    top_n: int,
    cache_dir: Path,
    matcher: DeDoDeMatcher | None = None,
) -> np.ndarray:
    """Match DeDoDe only on MD top-N candidates; query flip only, gallery unflipped."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    sim_cache = _shortlist_similarity_cache_path(cache_dir, name, flip_query, top_n)
    if sim_cache.is_file():
        print(f"  DeDoDe N={top_n} cache hit {sim_cache}")
        return np.load(sim_cache)

    matcher = matcher or DeDoDeMatcher(device="cpu")
    resolved = resolve_image_paths(paths, data_root)
    t0 = time.time()
    shortlists = md_shortlists(md_sim, top_n)

    query_feats = load_or_build_feature_cache(
        name=name,
        paths=resolved,
        flip=flip_query,
        cache_dir=cache_dir,
        matcher=matcher,
    )
    db_feats = load_or_build_feature_cache(
        name=name,
        paths=resolved,
        flip=False,
        cache_dir=cache_dir,
        matcher=matcher,
    )
    sim = matcher.compute_shortlist_similarity(query_feats, db_feats, shortlists)
    np.save(sim_cache, sim)
    print(f"  DeDoDe N={top_n} flip={flip_query} done in {time.time() - t0:.1f}s -> {sim_cache}")
    return sim


def load_or_compute_dedode_similarity(
    *,
    name: str,
    data_root: Path,
    paths: Sequence[str],
    flip_query: bool,
    cache_dir: Path,
    matcher: DeDoDeMatcher | None = None,
) -> np.ndarray:
    """Build or load full DeDoDe match-count matrix; query flip only, gallery unflipped."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    sim_cache = _similarity_cache_path(cache_dir, name, flip_query)
    if sim_cache.is_file():
        print(f"  DeDoDe cache hit {sim_cache}")
        return np.load(sim_cache)

    matcher = matcher or DeDoDeMatcher(device="cpu")
    resolved = resolve_image_paths(paths, data_root)
    t0 = time.time()

    query_feats = load_or_build_feature_cache(
        name=name,
        paths=resolved,
        flip=flip_query,
        cache_dir=cache_dir,
        matcher=matcher,
    )
    db_feats = load_or_build_feature_cache(
        name=name,
        paths=resolved,
        flip=False,
        cache_dir=cache_dir,
        matcher=matcher,
    )
    sim = matcher.compute_similarity(query_feats, db_feats, ignore_diagonal=True)
    np.save(sim_cache, sim)
    print(f"  DeDoDe flip={flip_query} done in {time.time() - t0:.1f}s -> {sim_cache}")
    return sim
