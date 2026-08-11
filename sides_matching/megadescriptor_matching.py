"""MegaDescriptor embedding extraction with optional YOLO kepala preprocessing."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics.pairwise import cosine_similarity

from sides_matching.loma_matching import resolve_image_paths
from sides_matching.utils import get_transform
from sides_matching.yolo_kepala_preprocessing import (
    KepalaCropper,
    load_rgb_pil,
    preprocess_tag,
)

MODEL_NAME = "hf-hub:BVRA/MegaDescriptor-L-384"


def md_feature_cache_path(
    cache_dir: Path,
    name: str,
    flip: bool,
    *,
    preprocess: str,
) -> Path:
    return cache_dir / f"{name}_md_{preprocess}_flip{flip}_feats.npy"


def md_similarity_cache_path(
    cache_dir: Path,
    name: str,
    *,
    preprocess: str,
    flip_query: bool,
) -> Path:
    return cache_dir / f"{name}_md_{preprocess}_flip{flip_query}_sim.npy"


class MegaDescriptorExtractor:
    """Frozen MegaDescriptor-L-384 embeddings after shared kepala preprocessing."""

    def __init__(
        self,
        *,
        device: str | torch.device,
        img_size: int = 384,
        batch_size: int = 16,
        cropper: KepalaCropper | None = None,
    ) -> None:
        import timm

        self.device = torch.device(device)
        self.img_size = img_size
        self.batch_size = batch_size
        self.cropper = cropper
        self.preprocess = preprocess_tag(
            use_kepala=cropper is not None,
            min_area_fraction=cropper.min_area_fraction if cropper else 0.0,
        )
        self.transform = get_transform(
            flip=False,
            grayscale=False,
            img_size=img_size,
            normalize=True,
        )
        model = timm.create_model(MODEL_NAME, num_classes=0, pretrained=True)
        self.model = model.to(self.device).eval()

    def _pil_batch_tensors(self, paths: Sequence[Path], *, flip: bool) -> torch.Tensor:
        images = [load_rgb_pil(path, flip=flip, cropper=self.cropper) for path in paths]
        tensors = [self.transform(im) for im in images]
        return torch.stack(tensors, dim=0)

    @torch.inference_mode()
    def extract_features(self, paths: Sequence[Path], *, flip: bool) -> np.ndarray:
        feats: list[np.ndarray] = []
        for start in range(0, len(paths), self.batch_size):
            batch_paths = paths[start : start + self.batch_size]
            batch = self._pil_batch_tensors(batch_paths, flip=flip).to(self.device)
            emb = F.normalize(self.model(batch), dim=1)
            feats.append(emb.cpu().numpy())
        return np.concatenate(feats, axis=0)


def load_or_build_md_features(
    *,
    name: str,
    paths: Sequence[Path],
    flip: bool,
    cache_dir: Path,
    extractor: MegaDescriptorExtractor,
) -> np.ndarray:
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache = md_feature_cache_path(
        cache_dir, name, flip, preprocess=extractor.preprocess
    )
    if cache.is_file():
        return np.load(cache)
    features = extractor.extract_features(paths, flip=flip)
    np.save(cache, features)
    return features


def load_or_compute_md_similarity(
    *,
    name: str,
    data_root: Path,
    paths: Sequence[str],
    flip_query: bool,
    cache_dir: Path,
    extractor: MegaDescriptorExtractor,
) -> np.ndarray:
    """Cosine similarity with query flip only; gallery unflipped (same as pickles)."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    sim_cache = md_similarity_cache_path(
        cache_dir,
        name,
        preprocess=extractor.preprocess,
        flip_query=flip_query,
    )
    if sim_cache.is_file():
        print(f"  MD {extractor.preprocess} cache hit {sim_cache}")
        return np.load(sim_cache)

    resolved = resolve_image_paths(paths, data_root)
    t0 = time.time()
    query_feats = load_or_build_md_features(
        name=name,
        paths=resolved,
        flip=flip_query,
        cache_dir=cache_dir,
        extractor=extractor,
    )
    db_feats = load_or_build_md_features(
        name=name,
        paths=resolved,
        flip=False,
        cache_dir=cache_dir,
        extractor=extractor,
    )
    sim = cosine_similarity(query_feats, db_feats).astype(np.float64)
    np.fill_diagonal(sim, -np.inf)
    np.save(sim_cache, sim)
    print(
        f"  MD {extractor.preprocess} flip={flip_query} "
        f"done in {time.time() - t0:.1f}s -> {sim_cache}"
    )
    return sim
