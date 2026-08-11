"""LoMa-B local feature matching for closed-set re-ID similarity matrices."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from PIL import Image, ImageOps
from tqdm import tqdm

from loma import LoMa, LoMaB
from loma.loma import filter_matches


@dataclass(frozen=True)
class LomaImageFeatures:
    keypoints: torch.Tensor
    descriptors: torch.Tensor
    height: int
    width: int

    def to_cpu_dict(self) -> dict[str, object]:
        return {
            "keypoints": self.keypoints.detach().cpu(),
            "descriptors": self.descriptors.detach().cpu(),
            "height": self.height,
            "width": self.width,
        }

    @classmethod
    def from_cpu_dict(cls, payload: dict[str, object]) -> LomaImageFeatures:
        return cls(
            keypoints=payload["keypoints"],
            descriptors=payload["descriptors"],
            height=int(payload["height"]),
            width=int(payload["width"]),
        )


def resolve_image_paths(paths: Sequence[str], data_root: Path) -> list[Path]:
    resolved: list[Path] = []
    for rel in paths:
        path = Path(rel)
        resolved.append(path if path.is_absolute() else data_root / path)
    return resolved


def _resize_dims(width: int, height: int, *, resize: int = 1024, patch: int = 14) -> tuple[int, int]:
    """Resize like LoMa DaD, but ensure DINOv2 patch divisibility."""
    scale = resize / max(width, height)
    width = int(scale * width)
    height = int(scale * height)
    width = max(patch, (width // patch) * patch)
    height = max(patch, (height // patch) * patch)
    return width, height


def _square_dim(size: int, *, patch: int = 14) -> int:
    """Largest patch-aligned square side ≤ size (DINOv2 requires % patch == 0)."""
    return max(patch, (size // patch) * patch)


def loma_resize_tag(*, resize: int | None, square_size: int | None) -> str:
    if square_size is not None:
        return f"sq{square_size}"
    if resize is not None:
        return f"max{resize}"
    return "native"


def load_loma_image(
    path: Path,
    *,
    flip: bool,
    device: torch.device,
    resize: int | None = 1024,
    square_size: int | None = None,
) -> torch.Tensor:
    """Load RGB image as LoMa tensor (1, 3, H, W) in [0, 1].

    Resize modes (first match wins):
    - square_size: force square (patch-aligned ≤ square_size)
    - resize: preserve aspect, max side ≈ resize (patch-aligned)
    - both None: no resize beyond patch alignment of native size
    """
    pil_im = Image.open(path).convert("RGB")
    if flip:
        pil_im = ImageOps.mirror(pil_im)
    if square_size is not None:
        side = _square_dim(square_size)
        pil_im = pil_im.resize((side, side))
    elif resize is not None:
        width, height = _resize_dims(*pil_im.size, resize=resize)
        pil_im = pil_im.resize((width, height))
    else:
        width, height = pil_im.size
        width = max(14, (width // 14) * 14)
        height = max(14, (height // 14) * 14)
        pil_im = pil_im.resize((width, height))
    arr = np.asarray(pil_im, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).to(device)[None]


class LomaMatcher:
    """Wrap LoMa-B with feature caching and match-count similarity."""

    def __init__(
        self,
        model: LoMa | None = None,
        *,
        resize: int | None = 1024,
        square_size: int | None = None,
        force_tensor_load: bool = False,
    ) -> None:
        self.model = model or LoMa(LoMaB())
        self.threshold = self.model.cfg.filter_threshold
        self.device = next(self.model.parameters()).device
        self.resize = resize
        self.square_size = square_size
        self.force_tensor_load = force_tensor_load or square_size is not None or resize != 1024
        self.resize_tag = loma_resize_tag(resize=resize, square_size=square_size)

    def extract_features(self, path: Path, *, flip: bool = False) -> LomaImageFeatures:
        # Default path: unflipped gallery uses LoMa's native path loader (DaD defaults).
        if flip or self.force_tensor_load:
            image = load_loma_image(
                path,
                flip=flip,
                device=self.device,
                resize=self.resize,
                square_size=self.square_size,
            )
            keypoints, descriptors, height, width = self.model.detect_and_describe(image)
        else:
            keypoints, descriptors, height, width = self.model.detect_and_describe(str(path))
        return LomaImageFeatures(keypoints, descriptors, height, width)

    def match_count(self, left: LomaImageFeatures, right: LomaImageFeatures) -> int:
        with torch.inference_mode():
            scores = self.model(
                left.keypoints,
                right.keypoints,
                left.descriptors,
                right.descriptors,
            )["scores"]
            matches, _, _, _ = filter_matches(scores, self.threshold)
        return int((matches[0] > -1).sum())

    def compute_similarity(
        self,
        query_features: Sequence[LomaImageFeatures],
        database_features: Sequence[LomaImageFeatures],
        *,
        ignore_diagonal: bool = True,
        show_progress: bool = True,
    ) -> np.ndarray:
        n_query = len(query_features)
        n_database = len(database_features)
        sim = np.zeros((n_query, n_database), dtype=np.float64)
        iterator = tqdm(range(n_query), desc="LoMa pairs", disable=not show_progress)
        for i in iterator:
            for j in range(n_database):
                if ignore_diagonal and n_query == n_database and i == j:
                    sim[i, j] = -np.inf
                    continue
                sim[i, j] = float(self.match_count(query_features[i], database_features[j]))
        return sim

    def compute_shortlist_similarity(
        self,
        query_features: Sequence[LomaImageFeatures],
        database_features: Sequence[LomaImageFeatures],
        shortlists: Sequence[Sequence[int]],
        *,
        show_progress: bool = True,
    ) -> np.ndarray:
        """Match each query only against MD shortlist indices (O(n·N))."""
        n_query = len(query_features)
        n_database = len(database_features)
        if len(shortlists) != n_query:
            raise ValueError("shortlists length must match number of queries")
        sim = np.full((n_query, n_database), -np.inf, dtype=np.float64)
        iterator = tqdm(range(n_query), desc="LoMa shortlist", disable=not show_progress)
        for i in iterator:
            for j in shortlists[i]:
                sim[i, int(j)] = float(
                    self.match_count(query_features[i], database_features[int(j)])
                )
        return sim


def _feature_cache_path(
    cache_dir: Path, name: str, flip: bool, *, resize_tag: str = "max1024"
) -> Path:
    if resize_tag == "max1024":
        return cache_dir / f"{name}_lomaB_flip{flip}_feats.pt"
    return cache_dir / f"{name}_lomaB_{resize_tag}_flip{flip}_feats.pt"


def _similarity_cache_path(
    cache_dir: Path, name: str, flip: bool, *, resize_tag: str = "max1024"
) -> Path:
    if resize_tag == "max1024":
        return cache_dir / f"{name}_lomaB_flip{flip}.npy"
    return cache_dir / f"{name}_lomaB_{resize_tag}_flip{flip}.npy"


def _shortlist_similarity_cache_path(
    cache_dir: Path, name: str, flip: bool, top_n: int, *, resize_tag: str = "max1024"
) -> Path:
    if resize_tag == "max1024":
        return cache_dir / f"{name}_lomaB_flip{flip}_N{top_n}.npy"
    return cache_dir / f"{name}_lomaB_{resize_tag}_flip{flip}_N{top_n}.npy"


def md_shortlists(md_sim: np.ndarray, top_n: int) -> list[np.ndarray]:
    shortlists: list[np.ndarray] = []
    for i in range(md_sim.shape[0]):
        finite = np.flatnonzero(np.isfinite(md_sim[i]))
        order = finite[np.argsort(-md_sim[i, finite])]
        shortlists.append(order[:top_n])
    return shortlists


def load_or_build_feature_cache(
    *,
    name: str,
    paths: Sequence[Path],
    flip: bool,
    cache_dir: Path,
    matcher: LomaMatcher,
) -> list[LomaImageFeatures]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache = _feature_cache_path(cache_dir, name, flip, resize_tag=matcher.resize_tag)
    if cache.is_file():
        payloads = torch.load(cache, weights_only=False)
        return [LomaImageFeatures.from_cpu_dict(item) for item in payloads]

    features: list[LomaImageFeatures] = []
    for path in tqdm(paths, desc=f"LoMa features {matcher.resize_tag} flip={flip}"):
        features.append(matcher.extract_features(path, flip=flip))
    torch.save([feat.to_cpu_dict() for feat in features], cache)
    return features


def load_or_compute_loma_shortlist_similarity(
    *,
    name: str,
    data_root: Path,
    paths: Sequence[str],
    flip_query: bool,
    md_sim: np.ndarray,
    top_n: int,
    cache_dir: Path,
    matcher: LomaMatcher | None = None,
) -> np.ndarray:
    """Match LoMa only on MD top-N; query flip only, gallery unflipped."""
    matcher = matcher or LomaMatcher()
    cache_dir.mkdir(parents=True, exist_ok=True)
    sim_cache = _shortlist_similarity_cache_path(
        cache_dir, name, flip_query, top_n, resize_tag=matcher.resize_tag
    )
    if sim_cache.is_file():
        print(f"  LoMa {matcher.resize_tag} N={top_n} cache hit {sim_cache}")
        return np.load(sim_cache)

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
        f"  LoMa {matcher.resize_tag} N={top_n} flip={flip_query} "
        f"done in {time.time() - t0:.1f}s -> {sim_cache}"
    )
    return sim


def load_or_compute_loma_similarity(
    *,
    name: str,
    data_root: Path,
    paths: Sequence[str],
    flip_query: bool,
    cache_dir: Path,
    matcher: LomaMatcher | None = None,
) -> np.ndarray:
    """Build or load LoMa match-count matrix; query flip only, gallery unflipped."""
    matcher = matcher or LomaMatcher()
    cache_dir.mkdir(parents=True, exist_ok=True)
    sim_cache = _similarity_cache_path(
        cache_dir, name, flip_query, resize_tag=matcher.resize_tag
    )
    if sim_cache.is_file():
        print(f"  LoMa {matcher.resize_tag} cache hit {sim_cache}")
        return np.load(sim_cache)

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
    print(
        f"  LoMa {matcher.resize_tag} flip={flip_query} "
        f"done in {time.time() - t0:.1f}s -> {sim_cache}"
    )
    return sim
