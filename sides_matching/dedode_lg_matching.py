"""DeDoDe v2 + LightGlue matching for MD shortlist re-ID (O(n·N)).

Uses Kornia DeDoDe detector L-C4-v2 + descriptor B with LightGlue(dedodeb).
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from kornia.feature import DeDoDe, LightGlue
from PIL import Image, ImageOps
from tqdm import tqdm

from sides_matching.loma_matching import resolve_image_paths
from sides_matching.xfeat_matching import md_shortlists


@dataclass(frozen=True)
class DeDoDeLGFeatures:
    keypoints: torch.Tensor  # (1, K, 2)
    descriptors: torch.Tensor  # (1, K, D)
    image_size: tuple[int, int]  # (W, H)

    def to_cpu_dict(self) -> dict[str, object]:
        return {
            "keypoints": self.keypoints.detach().cpu(),
            "descriptors": self.descriptors.detach().cpu(),
            "image_size": self.image_size,
        }

    @classmethod
    def from_cpu_dict(cls, payload: dict[str, object]) -> DeDoDeLGFeatures:
        return cls(
            keypoints=payload["keypoints"],
            descriptors=payload["descriptors"],
            image_size=(int(payload["image_size"][0]), int(payload["image_size"][1])),
        )


def load_rgb_square(path: Path, *, flip: bool, size: int) -> torch.Tensor:
    pil_im = Image.open(path).convert("RGB")
    if flip:
        pil_im = ImageOps.mirror(pil_im)
    pil_im = pil_im.resize((size, size))
    arr = np.asarray(pil_im, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1)[None]


class DeDoDeLightGlueMatcher:
    """DeDoDe v2 detector + B descriptor + LightGlue match-count similarity."""

    def __init__(
        self,
        *,
        device: str | torch.device,
        num_keypoints: int = 1024,
        image_size: int = 784,
        detector_weights: str = "L-C4-v2",
        descriptor_weights: str = "B-upright",
        lightglue_features: str = "dedodeb",
    ) -> None:
        self.device = torch.device(device)
        self.num_keypoints = num_keypoints
        self.image_size = image_size
        self.dedode = (
            DeDoDe.from_pretrained(
                detector_weights=detector_weights,
                descriptor_weights=descriptor_weights,
            )
            .to(self.device)
            .eval()
        )
        self.lightglue = LightGlue(lightglue_features).to(self.device).eval()

    def extract_features(self, path: Path, *, flip: bool = False) -> DeDoDeLGFeatures:
        image = load_rgb_square(path, flip=flip, size=self.image_size).to(self.device)
        with torch.inference_mode():
            keypoints, _scores, descriptors = self.dedode(image, n=self.num_keypoints)
        return DeDoDeLGFeatures(
            keypoints=keypoints.detach().cpu(),
            descriptors=descriptors.detach().cpu(),
            image_size=(self.image_size, self.image_size),
        )

    def _lg_input(self, feat: DeDoDeLGFeatures) -> dict[str, torch.Tensor]:
        width, height = feat.image_size
        return {
            "keypoints": feat.keypoints.to(self.device),
            "descriptors": feat.descriptors.to(self.device),
            "image_size": torch.tensor(
                [[width, height]], device=self.device, dtype=torch.float32
            ),
        }

    def match_count(self, left: DeDoDeLGFeatures, right: DeDoDeLGFeatures) -> int:
        with torch.inference_mode():
            out = self.lightglue(
                {"image0": self._lg_input(left), "image1": self._lg_input(right)}
            )
        return int(out["matches"][0].shape[0])

    def compute_shortlist_similarity(
        self,
        query_features: Sequence[DeDoDeLGFeatures],
        database_features: Sequence[DeDoDeLGFeatures],
        shortlists: Sequence[Sequence[int]],
        *,
        show_progress: bool = True,
    ) -> np.ndarray:
        n_query = len(query_features)
        n_database = len(database_features)
        if len(shortlists) != n_query:
            raise ValueError("shortlists length must match number of queries")
        sim = np.full((n_query, n_database), -np.inf, dtype=np.float64)
        iterator = tqdm(
            range(n_query), desc="DeDoDe-v2+LG shortlist", disable=not show_progress
        )
        for i in iterator:
            for j in shortlists[i]:
                sim[i, int(j)] = float(
                    self.match_count(query_features[i], database_features[int(j)])
                )
        return sim


def _feature_cache_path(cache_dir: Path, name: str, flip: bool) -> Path:
    return cache_dir / f"{name}_dedodev2B_lg_flip{flip}_feats.pt"


def _shortlist_cache_path(cache_dir: Path, name: str, flip: bool, top_n: int) -> Path:
    return cache_dir / f"{name}_dedodev2B_lg_flip{flip}_N{top_n}.npy"


def load_or_build_feature_cache(
    *,
    name: str,
    paths: Sequence[Path],
    flip: bool,
    cache_dir: Path,
    matcher: DeDoDeLightGlueMatcher,
) -> list[DeDoDeLGFeatures]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache = _feature_cache_path(cache_dir, name, flip)
    if cache.is_file():
        payloads = torch.load(cache, weights_only=False)
        return [DeDoDeLGFeatures.from_cpu_dict(item) for item in payloads]

    features: list[DeDoDeLGFeatures] = []
    for path in tqdm(paths, desc=f"DeDoDe-v2+LG features flip={flip}"):
        features.append(matcher.extract_features(path, flip=flip))
    torch.save([feat.to_cpu_dict() for feat in features], cache)
    return features


def load_or_compute_dedode_lg_shortlist_similarity(
    *,
    name: str,
    data_root: Path,
    paths: Sequence[str],
    flip_query: bool,
    md_sim: np.ndarray,
    top_n: int,
    cache_dir: Path,
    matcher: DeDoDeLightGlueMatcher | None = None,
) -> np.ndarray:
    """Match DeDoDe-v2+LG only on MD top-N; query flip only, gallery unflipped."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    sim_cache = _shortlist_cache_path(cache_dir, name, flip_query, top_n)
    if sim_cache.is_file():
        print(f"  DeDoDe-v2+LG N={top_n} cache hit {sim_cache}")
        return np.load(sim_cache)

    matcher = matcher or DeDoDeLightGlueMatcher(device="cpu")
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
    print(
        f"  DeDoDe-v2+LG N={top_n} flip={flip_query} done in {time.time() - t0:.1f}s "
        f"-> {sim_cache}"
    )
    return sim
