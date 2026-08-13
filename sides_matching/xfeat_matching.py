"""XFeat local feature matching for closed-set re-ID similarity matrices.

Uses Kornia's XFeat (CVPR 2024): https://arxiv.org/abs/2404.19174
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from kornia.feature import XFeat
from PIL import Image
from tqdm import tqdm

from sides_matching.loma_matching import resolve_image_paths
from sides_matching.yolo_kepala_preprocessing import (
    KepalaCropper,
    load_rgb_pil,
    preprocess_tag,
)

XFEAT_WEIGHTS_URL = "https://github.com/verlab/accelerated_features/raw/main/weights/xfeat.pt"
XFEAT_WEIGHTS_PATH = Path.home() / ".cache" / "torch" / "hub" / "checkpoints" / "xfeat.pt"


@dataclass(frozen=True)
class XFeatImageFeatures:
    keypoints: torch.Tensor
    descriptors: torch.Tensor
    scores: torch.Tensor

    def to_cpu_dict(self) -> dict[str, object]:
        return {
            "keypoints": self.keypoints.detach().cpu(),
            "descriptors": self.descriptors.detach().cpu(),
            "scores": self.scores.detach().cpu(),
        }

    @classmethod
    def from_cpu_dict(cls, payload: dict[str, object]) -> XFeatImageFeatures:
        return cls(
            keypoints=payload["keypoints"],
            descriptors=payload["descriptors"],
            scores=payload["scores"],
        )


def load_xfeat_weights(model: XFeat) -> XFeat:
    """Load official XFeat weights (curl cache preferred; avoids SSL hub failures)."""
    if XFEAT_WEIGHTS_PATH.is_file():
        state = torch.load(XFEAT_WEIGHTS_PATH, map_location="cpu", weights_only=True)
        model.net.load_state_dict(state)
        return model
    state = torch.hub.load_state_dict_from_url(XFEAT_WEIGHTS_URL, file_name="xfeat.pt")
    model.net.load_state_dict(state)
    return model


def load_xfeat_image(
    path: Path,
    *,
    flip: bool,
    device: torch.device,
    max_size: int | None = 800,
    square_size: int | None = None,
    cropper: KepalaCropper | None = None,
    bbox=None,
) -> torch.Tensor:
    """Load RGB image as (1, 3, H, W) float in [0, 1].

    Preprocessing order: open → dataset bbox (optional) → flip → YOLO crop → resize.

    Resize modes (first match wins):
    - square_size: force (square_size, square_size)
    - max_size: preserve aspect, max side = max_size
    - both None: no resize (native resolution)
    """
    pil_im = load_rgb_pil(path, flip=flip, cropper=cropper, bbox=bbox)
    if square_size is not None:
        pil_im = pil_im.resize((square_size, square_size))
    elif max_size is not None:
        width, height = pil_im.size
        scale = max_size / max(width, height)
        pil_im = pil_im.resize((max(1, int(width * scale)), max(1, int(height * scale))))
    arr = np.asarray(pil_im, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).to(device)[None]


def xfeat_resize_tag(*, max_size: int | None, square_size: int | None) -> str:
    if square_size is not None:
        return f"sq{square_size}"
    if max_size is not None:
        return f"max{max_size}"
    return "native"


def xfeat_cache_tag(
    *,
    max_size: int | None,
    square_size: int | None,
    use_kepala: bool,
    min_area_fraction: float = 0.0,
    pad_fraction: float = 0.0,
    use_bbox: bool = False,
) -> str:
    resize = xfeat_resize_tag(max_size=max_size, square_size=square_size)
    pre = preprocess_tag(
        use_kepala=use_kepala,
        min_area_fraction=min_area_fraction,
        pad_fraction=pad_fraction,
    )
    if use_bbox:
        resize = f"bbox_{resize}"
    if pre == "full":
        return resize
    return f"{pre}_{resize}"


class XFeatMatcher:
    """XFeat sparse detector/descriptor with MNN match-count similarity."""

    def __init__(
        self,
        *,
        device: str | torch.device,
        top_k: int = 4096,
        max_size: int | None = 800,
        square_size: int | None = None,
        min_cossim: float = 0.82,
        cropper: KepalaCropper | None = None,
        use_bbox: bool = False,
    ) -> None:
        self.device = torch.device(device)
        self.top_k = top_k
        self.max_size = max_size
        self.square_size = square_size
        self.min_cossim = min_cossim
        self.cropper = cropper
        self.use_bbox = use_bbox
        self.use_kepala = cropper is not None
        self.min_area_fraction = (
            cropper.min_area_fraction if cropper is not None else 0.0
        )
        self.pad_fraction = cropper.pad_fraction if cropper is not None else 0.0
        self.resize_tag = xfeat_cache_tag(
            max_size=max_size,
            square_size=square_size,
            use_kepala=self.use_kepala,
            min_area_fraction=self.min_area_fraction,
            pad_fraction=self.pad_fraction,
            use_bbox=use_bbox,
        )
        model = XFeat(top_k=top_k)
        load_xfeat_weights(model)
        self.model = model.to(self.device).eval()

    def extract_features(
        self, path: Path, *, flip: bool = False, bbox=None
    ) -> XFeatImageFeatures:
        image = load_xfeat_image(
            path,
            flip=flip,
            device=self.device,
            max_size=self.max_size,
            square_size=self.square_size,
            cropper=self.cropper,
            bbox=bbox,
        )
        with torch.inference_mode():
            out = self.model.detectAndCompute(image, top_k=self.top_k)[0]
        return XFeatImageFeatures(
            keypoints=out["keypoints"].detach().cpu(),
            descriptors=out["descriptors"].detach().cpu(),
            scores=out["scores"].detach().cpu(),
        )

    def match_count(self, left: XFeatImageFeatures, right: XFeatImageFeatures) -> int:
        if left.descriptors.numel() == 0 or right.descriptors.numel() == 0:
            return 0
        desc_a = left.descriptors.to(self.device)
        desc_b = right.descriptors.to(self.device)
        with torch.inference_mode():
            idx0, _ = self.model._match_mnn(desc_a, desc_b, min_cossim=self.min_cossim)
        return int(len(idx0))

    def compute_shortlist_similarity(
        self,
        query_features: Sequence[XFeatImageFeatures],
        database_features: Sequence[XFeatImageFeatures],
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
        iterator = tqdm(range(n_query), desc="XFeat shortlist", disable=not show_progress)
        for i in iterator:
            for j in shortlists[i]:
                sim[i, int(j)] = float(
                    self.match_count(query_features[i], database_features[int(j)])
                )
        return sim


def _feature_cache_path(
    cache_dir: Path, name: str, flip: bool, *, resize_tag: str = "max800"
) -> Path:
    # Keep legacy filename for the historical default (max side 800).
    if resize_tag == "max800":
        return cache_dir / f"{name}_xfeat_flip{flip}_feats.pt"
    return cache_dir / f"{name}_xfeat_{resize_tag}_flip{flip}_feats.pt"


def _shortlist_similarity_cache_path(
    cache_dir: Path,
    name: str,
    flip: bool,
    top_n: int,
    *,
    resize_tag: str = "max800",
    md_preprocess: str = "full",
) -> Path:
    md_part = f"_md{md_preprocess}" if md_preprocess != "full" else ""
    if resize_tag == "max800":
        return cache_dir / f"{name}_xfeat_flip{flip}{md_part}_N{top_n}.npy"
    return cache_dir / f"{name}_xfeat_{resize_tag}{md_part}_flip{flip}_N{top_n}.npy"


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
    matcher: XFeatMatcher,
    bboxes: Sequence | None = None,
) -> list[XFeatImageFeatures]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache = _feature_cache_path(cache_dir, name, flip, resize_tag=matcher.resize_tag)
    if cache.is_file():
        payloads = torch.load(cache, weights_only=False)
        return [XFeatImageFeatures.from_cpu_dict(item) for item in payloads]

    if bboxes is None:
        bboxes = [None] * len(paths)
    if len(bboxes) != len(paths):
        raise ValueError("bboxes length must match paths")
    features: list[XFeatImageFeatures] = []
    for path, bbox in tqdm(
        zip(paths, bboxes, strict=True),
        total=len(paths),
        desc=f"XFeat features {matcher.resize_tag} flip={flip}",
    ):
        features.append(matcher.extract_features(path, flip=flip, bbox=bbox))
    torch.save([feat.to_cpu_dict() for feat in features], cache)
    return features


def load_or_compute_xfeat_shortlist_similarity(
    *,
    name: str,
    data_root: Path,
    paths: Sequence[str],
    flip_query: bool,
    md_sim: np.ndarray,
    top_n: int,
    cache_dir: Path,
    matcher: XFeatMatcher | None = None,
    bboxes: Sequence | None = None,
) -> np.ndarray:
    """Match XFeat only on MD top-N candidates; query flip only, gallery unflipped."""
    matcher = matcher or XFeatMatcher(device="cpu")
    cache_dir.mkdir(parents=True, exist_ok=True)
    md_preprocess = preprocess_tag(
        use_kepala=matcher.use_kepala,
        min_area_fraction=matcher.min_area_fraction,
        pad_fraction=matcher.pad_fraction,
    )
    sim_cache = _shortlist_similarity_cache_path(
        cache_dir,
        name,
        flip_query,
        top_n,
        resize_tag=matcher.resize_tag,
        md_preprocess=md_preprocess,
    )
    if sim_cache.is_file():
        print(f"  XFeat {matcher.resize_tag} N={top_n} cache hit {sim_cache}")
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
        bboxes=bboxes,
    )
    db_feats = load_or_build_feature_cache(
        name=name,
        paths=resolved,
        flip=False,
        cache_dir=cache_dir,
        matcher=matcher,
        bboxes=bboxes,
    )
    sim = matcher.compute_shortlist_similarity(query_feats, db_feats, shortlists)
    np.save(sim_cache, sim)
    print(
        f"  XFeat {matcher.resize_tag} N={top_n} flip={flip_query} "
        f"done in {time.time() - t0:.1f}s -> {sim_cache}"
    )
    return sim
