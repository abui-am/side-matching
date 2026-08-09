"""LoRA fine-tuning helpers for MegaDescriptor with ArcFace metric learning."""

from __future__ import annotations

import csv
import json
import math
import os
import random
from dataclasses import asdict, dataclass, field
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from peft import LoraConfig, PeftModel, get_peft_model
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from wildlife_datasets.datasets import WildlifeDataset

from sides_matching.evaluation import (
    collect_embeddings,
    compute_retrieval_result,
    macro_average,
    normalize_orientation_codes,
    opposite_embedding_recall_at_k,
    verify_adapter_active,
)

MODEL_NAME = "hf-hub:BVRA/MegaDescriptor-L-384"
DEFAULT_LORA_TARGETS = ("qkv", "proj", "fc1", "fc2")
REUNION_SOURCES = ("ReunionGreen", "ReunionHawksbill")


@dataclass
class TrainConfig:
    model_name: str = MODEL_NAME
    img_size: int = 384
    batch_size: int = 1
    epochs: int = 40
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: Tuple[str, ...] = DEFAULT_LORA_TARGETS
    lr_lora: float = 2e-4
    lr_head: float = 1e-3
    weight_decay: float = 0.01
    arcface_s: float = 30.0
    arcface_m: float = 0.5
    opposite_loss_weight: float = 1.0
    triplet_margin: float = 0.2
    triplet_loss_weight: float = 1.0
    loss_mode: str = "triplet_hard"
    val_identity_fraction: float = 0.2
    early_stop_patience: int = 10
    checkpoint_metric: str = "macro_opp_recall"
    augment_policy: str = "standard"
    seed: int = 42
    use_amp: bool = True
    grad_checkpointing: bool = True
    num_workers: int = 2
    hard_negative_top_k: int = 50


@dataclass
class AugmentPolicy:
    name: str = "standard"
    rotation_deg: float = 5.0
    crop_scale: Tuple[float, float] = (0.85, 1.0)
    color_jitter: Tuple[float, float, float, float] = (0.2, 0.2, 0.1, 0.02)
    flip_p: float = 0.5
    blur_p: float = 0.3
    blur_kernel_range: Tuple[int, int] = (3, 5)
    grayscale_p: float = 0.0


AUGMENT_POLICIES: Dict[str, AugmentPolicy] = {
    "minimal": AugmentPolicy(
        name="minimal",
        flip_p=0.0,
        blur_p=0.0,
        grayscale_p=0.0,
    ),
    "standard": AugmentPolicy(
        name="standard",
        flip_p=0.5,
        blur_p=0.3,
        blur_kernel_range=(3, 5),
    ),
    "strong": AugmentPolicy(
        name="strong",
        flip_p=0.5,
        blur_p=0.5,
        blur_kernel_range=(5, 9),
        color_jitter=(0.3, 0.3, 0.15, 0.03),
        grayscale_p=0.05,
    ),
}


def get_augment_policy(name: str) -> AugmentPolicy:
    if name not in AUGMENT_POLICIES:
        raise ValueError(f"Unknown augment policy {name!r}; choose from {sorted(AUGMENT_POLICIES)}")
    return AUGMENT_POLICIES[name]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class ArcFaceHead(nn.Module):
    """ArcFace classification head for metric learning."""

    def __init__(self, embedding_dim: int, num_classes: int, s: float = 30.0, m: float = 0.3) -> None:
        super().__init__()
        self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.empty(num_classes, embedding_dim))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        embeddings = F.normalize(embeddings, dim=1)
        weight = F.normalize(self.weight, dim=1)
        cosine = F.linear(embeddings, weight)
        cosine = cosine.clamp(-1.0 + 1e-7, 1.0 - 1e-7)
        theta = torch.acos(cosine)
        target_logits = torch.cos(theta + self.m)
        one_hot = F.one_hot(labels, num_classes=weight.size(0)).float()
        logits = cosine * (1.0 - one_hot) + target_logits * one_hot
        return logits * self.s


class MegaDescriptorLoRA(nn.Module):
    """MegaDescriptor backbone with optional ArcFace head for training."""

    def __init__(self, backbone: nn.Module, arcface_head: Optional[ArcFaceHead] = None) -> None:
        super().__init__()
        self.backbone = backbone
        self.arcface_head = arcface_head

    def encode(self, images: torch.Tensor) -> torch.Tensor:
        return self.backbone(images)

    def forward(self, images: torch.Tensor, labels: Optional[torch.Tensor] = None) -> torch.Tensor:
        embeddings = self.encode(images)
        if self.arcface_head is None or labels is None:
            return embeddings
        return self.arcface_head(embeddings, labels)


class IdentityMapper:
    """Map dataset-prefixed identities to contiguous class indices."""

    def __init__(self, identity_to_label: Optional[Dict[str, int]] = None) -> None:
        self.identity_to_label = identity_to_label or {}
        self.label_to_identity = {v: k for k, v in self.identity_to_label.items()}

    @classmethod
    def from_identities(cls, identities: Iterable[str]) -> "IdentityMapper":
        unique = sorted(set(identities))
        return cls({identity: idx for idx, identity in enumerate(unique)})

    def encode(self, identity: str) -> int:
        return self.identity_to_label[identity]

    def encode_series(self, identities: pd.Series) -> np.ndarray:
        return identities.map(self.encode).to_numpy(dtype=np.int64)

    @property
    def num_classes(self) -> int:
        return len(self.identity_to_label)

    def to_dict(self) -> Dict[str, int]:
        return dict(self.identity_to_label)


def ensure_orientation_column(df: pd.DataFrame) -> pd.DataFrame:
    """Guarantee an orientation column (left/right) exists for opposite-side training."""
    if "orientation" in df.columns and df["orientation"].notna().any():
        return df
    out = df.copy()
    source = None
    for column in ("path", "image", "image_name", "file", "filename"):
        if column in out.columns:
            source = out[column].astype(str)
            break
    if source is None:
        raise KeyError(
            "Could not find orientation or a path/image column to infer left/right from."
        )
    text = source.str.lower()
    orientation = np.where(
        text.str.contains("left") | text.str.contains(r"(?:^|[_/-])l(?:[_/-]|$)"),
        "left",
        np.where(
            text.str.contains("right") | text.str.contains(r"(?:^|[_/-])r(?:[_/-]|$)"),
            "right",
            "unknown",
        ),
    )
    out["orientation"] = orientation
    return out


def prefixed_identity(dataset_name: str, identity: str) -> str:
    return f"{dataset_name}:{identity}"


def merge_train_dataframes(
    dataset_frames: Sequence[Tuple[str, pd.DataFrame]],
) -> pd.DataFrame:
    """Merge catalogues from multiple datasets with prefixed identities."""
    frames = []
    for dataset_name, df in dataset_frames:
        part = df.copy()
        part["global_identity"] = part["identity"].astype(str).map(
            lambda x: prefixed_identity(dataset_name, x)
        )
        part["source_dataset"] = dataset_name
        frames.append(part)
    merged = pd.concat(frames, ignore_index=True)
    return merged.reset_index(drop=True)


def split_identities(
    df: pd.DataFrame,
    identity_col: str = "global_identity",
    val_fraction: float = 0.2,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split by identity so the same turtle never appears in both train and val."""
    rng = np.random.default_rng(seed)
    identities = np.array(sorted(df[identity_col].unique()), dtype=object)
    rng.shuffle(identities)
    n_val = max(1, int(round(len(identities) * val_fraction)))
    val_ids = set(identities[:n_val])
    train_ids = set(identities[n_val:])
    train_df = df[df[identity_col].isin(train_ids)].reset_index(drop=True)
    val_df = df[df[identity_col].isin(val_ids)].reset_index(drop=True)
    return train_df, val_df


def split_identities_stratified(
    df: pd.DataFrame,
    val_fraction: float = 0.2,
    seed: int = 42,
    identity_col: str = "global_identity",
    source_col: str = "source_dataset",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Identity-disjoint train/val split stratified by source dataset."""
    train_parts: List[pd.DataFrame] = []
    val_parts: List[pd.DataFrame] = []
    for offset, source in enumerate(sorted(df[source_col].unique())):
        subset = df[df[source_col] == source].reset_index(drop=True)
        train_part, val_part = split_identities(
            subset,
            identity_col=identity_col,
            val_fraction=val_fraction,
            seed=seed + offset * 1000,
        )
        train_parts.append(train_part)
        val_parts.append(val_part)
    return (
        pd.concat(train_parts, ignore_index=True),
        pd.concat(val_parts, ignore_index=True),
    )


def location_holdout_split(
    merged_df: pd.DataFrame,
    holdout_sources: Sequence[str],
    val_fraction: float = 0.2,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Hold out entire source location(s) as pseudo-target; split remaining for train/val."""
    holdout = merged_df[merged_df["source_dataset"].isin(holdout_sources)].reset_index(drop=True)
    train_pool = merged_df[~merged_df["source_dataset"].isin(holdout_sources)].reset_index(drop=True)
    train_df, val_df = split_identities_stratified(train_pool, val_fraction=val_fraction, seed=seed)
    return train_df, val_df, holdout


class PairedAugmentTransform:
    """Shared geometric/color/blur augmentation for anchor-positive pairs."""

    def __init__(self, policy: AugmentPolicy, img_size: int) -> None:
        self.policy = policy
        self.img_size = img_size
        self.normalize = T.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        )

    def _odd_kernel(self, rng: np.random.Generator) -> int:
        low, high = self.policy.blur_kernel_range
        kernel = int(rng.integers(low, high + 1))
        if kernel % 2 == 0:
            kernel += 1
        return kernel

    def _draw_params(self, rng: np.random.Generator) -> Dict[str, object]:
        scale_min, scale_max = self.policy.crop_scale
        crop_scale = float(rng.uniform(scale_min, scale_max))
        crop_size = max(1, int(round(self.img_size * crop_scale)))
        max_offset = max(0, self.img_size - crop_size)
        top = int(rng.integers(0, max_offset + 1)) if max_offset else 0
        left = int(rng.integers(0, max_offset + 1)) if max_offset else 0
        jitter = self.policy.color_jitter
        return {
            "flip": bool(rng.random() < self.policy.flip_p),
            "angle": float(rng.uniform(-self.policy.rotation_deg, self.policy.rotation_deg)),
            "crop_size": crop_size,
            "top": top,
            "left": left,
            "blur": bool(rng.random() < self.policy.blur_p),
            "blur_kernel": self._odd_kernel(rng),
            "brightness": float(rng.uniform(max(0.0, 1.0 - jitter[0]), 1.0 + jitter[0])),
            "contrast": float(rng.uniform(max(0.0, 1.0 - jitter[1]), 1.0 + jitter[1])),
            "saturation": float(rng.uniform(max(0.0, 1.0 - jitter[2]), 1.0 + jitter[2])),
            "hue": float(rng.uniform(-jitter[3], jitter[3])),
            "grayscale": bool(rng.random() < self.policy.grayscale_p),
        }

    def _apply(self, image: Image.Image, params: Dict[str, object]) -> torch.Tensor:
        img = image.convert("RGB")
        img = TF.resize(img, [self.img_size, self.img_size])
        crop_size = int(params["crop_size"])
        img = TF.crop(img, int(params["top"]), int(params["left"]), crop_size, crop_size)
        img = TF.resize(img, [self.img_size, self.img_size])
        if params["flip"]:
            img = TF.hflip(img)
        img = TF.rotate(img, float(params["angle"]))
        if params["grayscale"]:
            img = TF.rgb_to_grayscale(img, num_output_channels=3)
        img = TF.adjust_brightness(img, float(params["brightness"]))
        img = TF.adjust_contrast(img, float(params["contrast"]))
        img = TF.adjust_saturation(img, float(params["saturation"]))
        img = TF.adjust_hue(img, float(params["hue"]))
        if params["blur"]:
            img = TF.gaussian_blur(img, kernel_size=[int(params["blur_kernel"]), int(params["blur_kernel"])])
        tensor = TF.to_tensor(img)
        return self.normalize(tensor)

    def apply_pair(
        self,
        anchor: Image.Image,
        positive: Image.Image,
        rng: np.random.Generator,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        params = self._draw_params(rng)
        return self._apply(anchor, params), self._apply(positive, params)

    def apply_single(self, image: Image.Image, rng: np.random.Generator) -> torch.Tensor:
        params = self._draw_params(rng)
        return self._apply(image, params)


def get_train_transform(img_size: int = 384, policy_name: str = "minimal") -> T.Compose:
    """Legacy single-image transform; prefer PairedAugmentTransform for training."""
    policy = get_augment_policy(policy_name)
    transforms: List[nn.Module] = [T.Resize([img_size, img_size])]
    if policy.flip_p > 0:
        transforms.append(T.RandomHorizontalFlip(policy.flip_p))
    transforms.append(
        T.ColorJitter(
            brightness=policy.color_jitter[0],
            contrast=policy.color_jitter[1],
            saturation=policy.color_jitter[2],
            hue=policy.color_jitter[3],
        )
    )
    if policy.blur_p > 0:
        kernel = policy.blur_kernel_range[1]
        transforms.append(T.RandomApply([T.GaussianBlur(kernel, sigma=(0.1, 2.0))], p=policy.blur_p))
    transforms.extend([
        T.ToTensor(),
        T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])
    return T.Compose(transforms)


def get_eval_transform(flip: bool = False, img_size: int = 384) -> T.Compose:
    transforms: List[nn.Module] = [T.Resize([img_size, img_size])]
    if flip:
        transforms.append(T.RandomHorizontalFlip(1))
    transforms.extend([
        T.ToTensor(),
        T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])
    return T.Compose(transforms)


class RawConcatLabelledDataset(Dataset):
    """Concatenate WildlifeDataset instances returning raw PIL images."""

    def __init__(
        self,
        datasets: Sequence[WildlifeDataset],
        label_parts: Sequence[np.ndarray],
    ) -> None:
        self.datasets = list(datasets)
        self.labels = np.concatenate(label_parts)
        self.cumulative = np.cumsum([0] + [len(dataset) for dataset in self.datasets])

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> Tuple[Image.Image, int]:
        for dataset_idx, start in enumerate(self.cumulative[:-1]):
            end = self.cumulative[dataset_idx + 1]
            if start <= index < end:
                image = self.datasets[dataset_idx][index - start]
                return image, int(self.labels[index])
        raise IndexError(index)


class ConcatLabelledDataset(Dataset):
    """Concatenate multiple WildlifeDataset instances with aligned labels."""

    def __init__(self, datasets: Sequence[WildlifeDataset], label_parts: Sequence[np.ndarray]) -> None:
        self.datasets = list(datasets)
        self.labels = np.concatenate(label_parts)
        self.cumulative = np.cumsum([0] + [len(dataset) for dataset in self.datasets])

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        for dataset_idx, start in enumerate(self.cumulative[:-1]):
            end = self.cumulative[dataset_idx + 1]
            if start <= index < end:
                image = self.datasets[dataset_idx][index - start]
                return image, torch.tensor(int(self.labels[index]), dtype=torch.long)
        raise IndexError(index)


class LabelledWildlifeDataset(Dataset):
    """PyTorch dataset over a WildlifeDataset catalogue with integer labels."""

    def __init__(
        self,
        wildlife_dataset: WildlifeDataset,
        labels: np.ndarray,
    ) -> None:
        self.wildlife_dataset = wildlife_dataset
        self.labels = labels

    def __len__(self) -> int:
        return len(self.wildlife_dataset)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        image = self.wildlife_dataset[index]
        label = int(self.labels[index])
        return image, torch.tensor(label, dtype=torch.long)


class HardNegativeMap:
    """Map training row index to a hard or random different-identity index."""

    def __init__(
        self,
        embeddings: np.ndarray,
        labels: np.ndarray,
        rng: np.random.Generator,
        top_k: int = 50,
    ) -> None:
        self.labels = labels.astype(np.int64)
        self.rng = rng
        self.top_k = top_k
        normalized = embeddings / np.maximum(np.linalg.norm(embeddings, axis=1, keepdims=True), 1e-8)
        similarity = normalized @ normalized.T
        np.fill_diagonal(similarity, -np.inf)
        self.negative_for_index: Dict[int, int] = {}
        unique_labels = np.unique(self.labels)
        label_to_indices: Dict[int, List[int]] = {}
        for idx, label in enumerate(self.labels.tolist()):
            label_to_indices.setdefault(int(label), []).append(idx)
        for index in range(len(self.labels)):
            order = np.argsort(-similarity[index])
            chosen = None
            for candidate in order[:top_k]:
                if self.labels[candidate] != self.labels[index]:
                    chosen = int(candidate)
                    break
            if chosen is None:
                other_labels = [label for label in unique_labels if label != self.labels[index]]
                if not other_labels:
                    chosen = index
                else:
                    other_label = int(self.rng.choice(other_labels))
                    chosen = int(self.rng.choice(label_to_indices[other_label]))
            self.negative_for_index[index] = chosen

    def get(self, index: int) -> int:
        return self.negative_for_index[index]


class RandomNegativeMap:
    """Pick a random different-identity index for each anchor."""

    def __init__(self, labels: np.ndarray, seed: int = 42) -> None:
        self.labels = labels.astype(np.int64)
        self.rng = np.random.default_rng(seed)
        self.negative_for_index: Dict[int, int] = {}
        label_to_indices: Dict[int, List[int]] = {}
        for idx, label in enumerate(self.labels.tolist()):
            label_to_indices.setdefault(int(label), []).append(idx)
        all_labels = list(label_to_indices.keys())
        for index in range(len(self.labels)):
            other_labels = [label for label in all_labels if label != self.labels[index]]
            other_label = int(self.rng.choice(other_labels))
            self.negative_for_index[index] = int(self.rng.choice(label_to_indices[other_label]))

    def get(self, index: int) -> int:
        return self.negative_for_index[index]


def build_hard_negative_map(
    embeddings: np.ndarray,
    labels: np.ndarray,
    seed: int = 42,
    top_k: int = 50,
) -> HardNegativeMap:
    return HardNegativeMap(embeddings, labels, np.random.default_rng(seed), top_k=top_k)


class OppositePairDataset(Dataset):
    """Sample (image_a, image_b, label) preferring opposite sides of the same ID."""

    def __init__(
        self,
        datasets: Sequence[WildlifeDataset],
        label_parts: Sequence[np.ndarray],
        orientation_parts: Sequence[np.ndarray],
        paired_augment: Optional[PairedAugmentTransform] = None,
        seed: int = 42,
    ) -> None:
        self.base = RawConcatLabelledDataset(datasets, label_parts)
        self.paired_augment = paired_augment
        self.orientations = np.concatenate([
            normalize_orientation_codes(part) for part in orientation_parts
        ])
        if len(self.orientations) != len(self.base):
            raise ValueError("orientation_parts length must match label_parts / datasets")
        self.rng = np.random.default_rng(seed)
        self.by_label: Dict[int, List[int]] = {}
        self.by_label_ori: Dict[Tuple[int, int], List[int]] = {}
        for index, label in enumerate(self.base.labels.tolist()):
            label_i = int(label)
            self.by_label.setdefault(label_i, []).append(index)
            ori = int(self.orientations[index])
            self.by_label_ori.setdefault((label_i, ori), []).append(index)

    def __len__(self) -> int:
        return len(self.base)

    def _sample_partner(self, index: int, label: int, ori: int) -> int:
        opposite = 1 - ori if ori in (0, 1) else -1
        if opposite in (0, 1):
            candidates = [
                j for j in self.by_label_ori.get((label, opposite), []) if j != index
            ]
            if candidates:
                return int(self.rng.choice(candidates))
        same_id = [j for j in self.by_label.get(label, []) if j != index]
        if same_id:
            return int(self.rng.choice(same_id))
        return index

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        image_a, label = self.base[index]
        partner = self._sample_partner(index, label, int(self.orientations[index]))
        image_b, _ = self.base[partner]
        if self.paired_augment is not None:
            image_a, image_b = self.paired_augment.apply_pair(image_a, image_b, self.rng)
        else:
            raise ValueError("OppositePairDataset requires paired_augment when using raw PIL images")
        return image_a, image_b, torch.tensor(label, dtype=torch.long)


class OppositeTripletDataset(Dataset):
    """Sample anchor, opposite-side positive, and different-identity negative."""

    def __init__(
        self,
        datasets: Sequence[WildlifeDataset],
        label_parts: Sequence[np.ndarray],
        orientation_parts: Sequence[np.ndarray],
        negative_map: HardNegativeMap,
        paired_augment: Optional[PairedAugmentTransform] = None,
        seed: int = 42,
    ) -> None:
        self.base = RawConcatLabelledDataset(datasets, label_parts)
        self.negative_map = negative_map
        self.paired_augment = paired_augment
        self.orientations = np.concatenate([
            normalize_orientation_codes(part) for part in orientation_parts
        ])
        if len(self.orientations) != len(self.base):
            raise ValueError("orientation_parts length must match label_parts / datasets")
        self.rng = np.random.default_rng(seed)
        self.by_label: Dict[int, List[int]] = {}
        self.by_label_ori: Dict[Tuple[int, int], List[int]] = {}
        for index, label in enumerate(self.base.labels.tolist()):
            label_i = int(label)
            self.by_label.setdefault(label_i, []).append(index)
            ori = int(self.orientations[index])
            self.by_label_ori.setdefault((label_i, ori), []).append(index)

    def __len__(self) -> int:
        return len(self.base)

    def _sample_partner(self, index: int, label: int, ori: int) -> int:
        opposite = 1 - ori if ori in (0, 1) else -1
        if opposite in (0, 1):
            candidates = [
                j for j in self.by_label_ori.get((label, opposite), []) if j != index
            ]
            if candidates:
                return int(self.rng.choice(candidates))
        same_id = [j for j in self.by_label.get(label, []) if j != index]
        if same_id:
            return int(self.rng.choice(same_id))
        return index

    def __getitem__(
        self,
        index: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        anchor_img, anchor_label = self.base[index]
        partner = self._sample_partner(index, anchor_label, int(self.orientations[index]))
        positive_img, _ = self.base[partner]
        negative_index = self.negative_map.get(index)
        negative_img, negative_label = self.base[negative_index]
        if negative_label == anchor_label:
            raise ValueError("Hard negative map returned same-identity negative")
        if self.paired_augment is None:
            raise ValueError("OppositeTripletDataset requires paired_augment")
        anchor_t, positive_t = self.paired_augment.apply_pair(anchor_img, positive_img, self.rng)
        negative_t = self.paired_augment.apply_single(negative_img, self.rng)
        return (
            anchor_t,
            positive_t,
            negative_t,
            torch.tensor(anchor_label, dtype=torch.long),
            torch.tensor(negative_label, dtype=torch.long),
        )


def build_concat_dataloader(
    datasets: Sequence[WildlifeDataset],
    label_parts: Sequence[np.ndarray],
    batch_size: int,
    shuffle: bool,
    num_workers: int = 2,
) -> DataLoader:
    dataset = ConcatLabelledDataset(datasets, label_parts)
    return DataLoader(
        dataset,
        batch_size=max(1, int(batch_size)),
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def build_opposite_pair_dataloader(
    datasets: Sequence[WildlifeDataset],
    label_parts: Sequence[np.ndarray],
    orientation_parts: Sequence[np.ndarray],
    batch_size: int,
    num_workers: int = 2,
    seed: int = 42,
    paired_augment: Optional[PairedAugmentTransform] = None,
) -> DataLoader:
    dataset = OppositePairDataset(
        datasets,
        label_parts,
        orientation_parts,
        paired_augment=paired_augment,
        seed=seed,
    )
    return DataLoader(
        dataset,
        batch_size=max(1, int(batch_size)),
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )


def build_opposite_triplet_dataloader(
    datasets: Sequence[WildlifeDataset],
    label_parts: Sequence[np.ndarray],
    orientation_parts: Sequence[np.ndarray],
    negative_map: HardNegativeMap,
    batch_size: int,
    num_workers: int = 2,
    seed: int = 42,
    paired_augment: Optional[PairedAugmentTransform] = None,
) -> DataLoader:
    dataset = OppositeTripletDataset(
        datasets,
        label_parts,
        orientation_parts,
        negative_map=negative_map,
        paired_augment=paired_augment,
        seed=seed,
    )
    return DataLoader(
        dataset,
        batch_size=max(1, int(batch_size)),
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )


def clear_cuda_memory() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def is_cuda_oom(exc: BaseException) -> bool:
    message = str(exc).lower()
    return isinstance(exc, torch.cuda.OutOfMemoryError) or (
        isinstance(exc, RuntimeError) and "out of memory" in message
    )


def enable_backbone_grad_checkpointing(backbone: nn.Module) -> bool:
    """Enable activation checkpointing on the timm / Peft-wrapped backbone if available."""
    candidates: List[nn.Module] = [backbone]
    get_base = getattr(backbone, "get_base_model", None)
    if callable(get_base) and not isinstance(get_base, nn.Module):
        candidates.append(get_base())
    for attr in ("base_model", "model"):
        value = getattr(backbone, attr, None)
        if isinstance(value, nn.Module):
            candidates.append(value)
    seen = set()
    for candidate in candidates:
        candidate_id = id(candidate)
        if candidate_id in seen:
            continue
        seen.add(candidate_id)
        if hasattr(candidate, "set_grad_checkpointing"):
            candidate.set_grad_checkpointing(True)
            return True
    return False


def build_dataloader(
    wildlife_dataset: WildlifeDataset,
    labels: np.ndarray,
    batch_size: int,
    shuffle: bool,
    num_workers: int = 2,
) -> DataLoader:
    dataset = LabelledWildlifeDataset(wildlife_dataset, labels)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def create_backbone(model_name: str = MODEL_NAME, pretrained: bool = True) -> nn.Module:
    import timm

    return timm.create_model(model_name, num_classes=0, pretrained=pretrained)


def attach_lora(
    model: nn.Module,
    r: int = 16,
    alpha: int = 32,
    dropout: float = 0.05,
    target_modules: Sequence[str] = DEFAULT_LORA_TARGETS,
) -> nn.Module:
    config = LoraConfig(
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=list(target_modules),
        bias="none",
    )
    return get_peft_model(model, config)


def build_training_model(
    num_classes: int,
    config: TrainConfig,
    device: torch.device,
) -> MegaDescriptorLoRA:
    clear_cuda_memory()
    backbone = create_backbone(config.model_name, pretrained=True)
    backbone = attach_lora(
        backbone,
        r=config.lora_r,
        alpha=config.lora_alpha,
        dropout=config.lora_dropout,
        target_modules=config.lora_target_modules,
    )
    if config.grad_checkpointing:
        enabled = enable_backbone_grad_checkpointing(backbone)
        print(f'Gradient checkpointing: {"on" if enabled else "unavailable"}')
    embedding_dim = backbone.num_features
    arcface_head = ArcFaceHead(
        embedding_dim=embedding_dim,
        num_classes=num_classes,
        s=config.arcface_s,
        m=config.arcface_m,
    )
    model = MegaDescriptorLoRA(backbone, arcface_head)
    return model.to(device)


def build_inference_model(
    config: TrainConfig,
    adapter_dir: str,
    device: torch.device,
) -> nn.Module:
    import timm

    backbone = timm.create_model(config.model_name, num_classes=0, pretrained=True)
    model = PeftModel.from_pretrained(backbone, adapter_dir, is_trainable=False)
    model = model.to(device)
    model.eval()
    return model


def configure_optimizer(model: MegaDescriptorLoRA, config: TrainConfig) -> torch.optim.Optimizer:
    lora_params = [p for p in model.backbone.parameters() if p.requires_grad]
    head_params = list(model.arcface_head.parameters())
    return torch.optim.AdamW(
        [
            {"params": lora_params, "lr": config.lr_lora},
            {"params": head_params, "lr": config.lr_head},
        ],
        weight_decay=config.weight_decay,
    )


def _triplet_cosine_loss(
    anchor: torch.Tensor,
    positive: torch.Tensor,
    negative: torch.Tensor,
    margin: float,
) -> torch.Tensor:
    sim_pos = (anchor * positive).sum(dim=-1)
    sim_neg = (anchor * negative).sum(dim=-1)
    return torch.relu(sim_neg - sim_pos + margin).mean()


def train_epoch(
    model: MegaDescriptorLoRA,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
    use_amp: bool = True,
    opposite_loss_weight: float = 0.0,
    triplet_margin: float = 0.2,
    triplet_loss_weight: float = 1.0,
) -> Dict[str, float]:
    """Train one epoch; supports pair or triplet batches."""
    model.train()
    totals = {
        "loss": 0.0,
        "loss_arc": 0.0,
        "loss_opp": 0.0,
        "loss_triplet": 0.0,
        "loss_neg": 0.0,
    }
    n_batches = 0
    amp_enabled = bool(use_amp and device.type == "cuda")
    for batch in loader:
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=amp_enabled):
            if len(batch) == 5:
                images_a, images_b, images_n, labels_a, labels_n = batch
                images_a = images_a.to(device, non_blocking=True)
                images_b = images_b.to(device, non_blocking=True)
                images_n = images_n.to(device, non_blocking=True)
                labels_a = labels_a.to(device, non_blocking=True)
                labels_n = labels_n.to(device, non_blocking=True)
                emb_a_raw = model.encode(images_a)
                emb_b_raw = model.encode(images_b)
                emb_n_raw = model.encode(images_n)
                emb_a = F.normalize(emb_a_raw, dim=1)
                emb_b = F.normalize(emb_b_raw, dim=1)
                emb_n = F.normalize(emb_n_raw, dim=1)
                logits_a = model.arcface_head(emb_a_raw, labels_a)
                logits_b = model.arcface_head(emb_b_raw, labels_a)
                logits_n = model.arcface_head(emb_n_raw, labels_n)
                loss_arc = (
                    F.cross_entropy(logits_a, labels_a)
                    + F.cross_entropy(logits_b, labels_a)
                    + F.cross_entropy(logits_n, labels_n)
                ) / 3.0
                loss_opp = (1.0 - (emb_a * emb_b).sum(dim=-1)).mean()
                loss_triplet = _triplet_cosine_loss(emb_a, emb_b, emb_n, triplet_margin)
                loss_neg = emb_a.new_zeros(())
                loss = (
                    loss_arc
                    + opposite_loss_weight * loss_opp
                    + triplet_loss_weight * loss_triplet
                )
            elif len(batch) == 3:
                images_a, images_b, labels = batch
                images_a = images_a.to(device, non_blocking=True)
                images_b = images_b.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                emb_a_raw = model.encode(images_a)
                emb_b_raw = model.encode(images_b)
                emb_a = F.normalize(emb_a_raw, dim=1)
                emb_b = F.normalize(emb_b_raw, dim=1)
                logits_a = model.arcface_head(emb_a_raw, labels)
                logits_b = model.arcface_head(emb_b_raw, labels)
                loss_arc = 0.5 * (
                    F.cross_entropy(logits_a, labels) + F.cross_entropy(logits_b, labels)
                )
                loss_opp = (1.0 - (emb_a * emb_b).sum(dim=-1)).mean()
                all_emb = torch.cat([emb_a, emb_b], dim=0)
                all_labels = torch.cat([labels, labels], dim=0)
                sim = all_emb @ all_emb.T
                same = all_labels.unsqueeze(0) == all_labels.unsqueeze(1)
                neg_mask = ~same
                if neg_mask.any():
                    loss_neg = torch.relu(sim[neg_mask]).mean()
                else:
                    loss_neg = emb_a.new_zeros(())
                loss_triplet = emb_a.new_zeros(())
                loss = loss_arc + opposite_loss_weight * (loss_opp + 0.5 * loss_neg)
            else:
                images, labels = batch
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                logits = model(images, labels)
                loss_arc = F.cross_entropy(logits, labels)
                loss_opp = loss_arc.new_zeros(())
                loss_triplet = loss_arc.new_zeros(())
                loss_neg = loss_arc.new_zeros(())
                loss = loss_arc
        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite training loss encountered")
        if scaler is not None and amp_enabled:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        totals["loss"] += float(loss.detach().item())
        totals["loss_arc"] += float(loss_arc.detach().item())
        totals["loss_opp"] += float(loss_opp.detach().item())
        totals["loss_triplet"] += float(loss_triplet.detach().item())
        totals["loss_neg"] += float(loss_neg.detach().item())
        n_batches += 1
    denom = max(n_batches, 1)
    return {key: value / denom for key, value in totals.items()}


@torch.no_grad()
def embedding_recall_at_1(
    model: MegaDescriptorLoRA,
    loader: DataLoader,
    device: torch.device,
    use_amp: bool = True,
) -> float:
    embeddings, labels = collect_embeddings(model, loader, device, use_amp=use_amp)
    if embeddings.numel() == 0:
        return 0.0
    result = compute_retrieval_result(embeddings, labels, opposite_only=False)
    return result.recall_at_1


@torch.no_grad()
def opposite_embedding_recall_at_1(
    model: MegaDescriptorLoRA,
    loader: DataLoader,
    orientations: np.ndarray,
    device: torch.device,
    use_amp: bool = True,
) -> float:
    embeddings, labels = collect_embeddings(model, loader, device, use_amp=use_amp)
    if embeddings.numel() == 0:
        return 0.0
    return opposite_embedding_recall_at_k(embeddings, labels, orientations, k=1)


@torch.no_grad()
def evaluate_source_validation(
    model: MegaDescriptorLoRA,
    source_loaders: Dict[str, DataLoader],
    source_orientations: Dict[str, np.ndarray],
    device: torch.device,
    use_amp: bool = True,
) -> Dict[str, Union[float, Dict[str, float]]]:
    source_embed: Dict[str, float] = {}
    source_opp: Dict[str, float] = {}
    for name, loader in source_loaders.items():
        embeddings, labels = collect_embeddings(model, loader, device, use_amp=use_amp)
        full = compute_retrieval_result(embeddings, labels, opposite_only=False)
        opp = compute_retrieval_result(
            embeddings,
            labels,
            orientations=source_orientations[name],
            opposite_only=True,
        )
        source_embed[name] = full.recall_at_1
        source_opp[name] = opp.recall_at_1
    return {
        "source_embed": source_embed,
        "source_opp": source_opp,
        "macro_embed_recall": macro_average(source_embed),
        "macro_opp_recall": macro_average(source_opp),
    }


@torch.no_grad()
def mine_train_baseline_embeddings(
    train_sources: Sequence[Tuple[str, str, Callable]],
    train_df: pd.DataFrame,
    identity_mapper: IdentityMapper,
    device: torch.device,
    config: TrainConfig,
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract frozen MegaDescriptor embeddings on source-train rows only."""
    import timm

    backbone = timm.create_model(config.model_name, num_classes=0, pretrained=True).to(device)
    backbone.eval()
    eval_transform = get_eval_transform(flip=False, img_size=config.img_size)
    train_datasets: List[WildlifeDataset] = []
    train_label_parts: List[np.ndarray] = []
    for name, root, dataset_fn in train_sources:
        source_train = train_df[train_df["source_dataset"] == name].reset_index(drop=True)
        if len(source_train) == 0:
            continue
        train_datasets.append(
            wildlife_dataset_from_df(root, source_train, dataset_fn, eval_transform)
        )
        train_label_parts.append(identity_mapper.encode_series(source_train["global_identity"]))
    loader = build_concat_dataloader(
        train_datasets,
        train_label_parts,
        batch_size=max(2, config.batch_size * 2),
        shuffle=False,
        num_workers=config.num_workers,
    )
    features: List[np.ndarray] = []
    labels: List[np.ndarray] = []
    for images, batch_labels in loader:
        images = images.to(device, non_blocking=True)
        batch = F.normalize(backbone(images), dim=1)
        features.append(batch.cpu().numpy())
        labels.append(batch_labels.numpy())
    del backbone
    clear_cuda_memory()
    if not features:
        return np.empty((0, 0), dtype=np.float32), np.empty((0,), dtype=np.int64)
    return np.concatenate(features, axis=0), np.concatenate(labels, axis=0)


def build_negative_map_for_training(
    config: TrainConfig,
    train_sources: Sequence[Tuple[str, str, Callable]],
    train_df: pd.DataFrame,
    identity_mapper: IdentityMapper,
    device: torch.device,
) -> Optional[Union[HardNegativeMap, RandomNegativeMap]]:
    if not config.loss_mode.startswith("triplet"):
        return None
    embeddings, labels = mine_train_baseline_embeddings(
        train_sources, train_df, identity_mapper, device, config
    )
    if config.loss_mode == "triplet_random":
        return RandomNegativeMap(labels, seed=config.seed)
    return build_hard_negative_map(
        embeddings, labels, seed=config.seed, top_k=config.hard_negative_top_k
    )


def build_train_loader(
    config: TrainConfig,
    train_datasets: Sequence[WildlifeDataset],
    train_label_parts: Sequence[np.ndarray],
    train_orientation_parts: Sequence[np.ndarray],
    negative_map: Optional[Union[HardNegativeMap, RandomNegativeMap]] = None,
) -> DataLoader:
    paired_augment = PairedAugmentTransform(
        get_augment_policy(config.augment_policy),
        img_size=config.img_size,
    )
    if config.loss_mode.startswith("triplet"):
        if negative_map is None:
            raise ValueError("triplet loss modes require a negative map")
        return build_opposite_triplet_dataloader(
            train_datasets,
            train_label_parts,
            train_orientation_parts,
            negative_map=negative_map,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
            seed=config.seed,
            paired_augment=paired_augment,
        )
    return build_opposite_pair_dataloader(
        train_datasets,
        train_label_parts,
        train_orientation_parts,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        seed=config.seed,
        paired_augment=paired_augment,
    )


def save_checkpoint(
    model: MegaDescriptorLoRA,
    identity_mapper: IdentityMapper,
    config: TrainConfig,
    checkpoint_dir: str,
    metrics: Optional[Dict[str, float]] = None,
) -> None:
    os.makedirs(checkpoint_dir, exist_ok=True)
    adapter_dir = os.path.join(checkpoint_dir, "adapter")
    model.backbone.save_pretrained(adapter_dir)
    torch.save(model.arcface_head.state_dict(), os.path.join(checkpoint_dir, "arcface_head.pt"))
    payload = {
        "config": asdict(config),
        "identity_to_label": identity_mapper.to_dict(),
        "metrics": metrics or {},
    }
    with open(os.path.join(checkpoint_dir, "train_config.json"), "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def append_metrics_row(csv_path: str, row: Dict[str, object]) -> None:
    directory = os.path.dirname(csv_path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    write_header = not os.path.exists(csv_path)
    fieldnames = list(row.keys())
    with open(csv_path, "a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def load_train_payload(checkpoint_dir: str) -> Dict:
    with open(os.path.join(checkpoint_dir, "train_config.json"), encoding="utf-8") as handle:
        return json.load(handle)


@torch.no_grad()
def extract_features(
    model: nn.Module,
    wildlife_dataset: WildlifeDataset,
    device: torch.device,
    batch_size: int = 32,
) -> np.ndarray:
    """Extract L2-normalized embeddings aligned with wildlife_dataset row order."""
    model.eval()
    loader = DataLoader(
        wildlife_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
    )
    features = []
    for images in loader:
        if isinstance(images, (tuple, list)):
            images = images[0]
        images = images.to(device, non_blocking=True)
        batch = F.normalize(model(images), dim=1)
        features.append(batch.cpu().numpy())
    if not features:
        num_features = getattr(model, "num_features", 1536)
        return np.empty((0, num_features), dtype=np.float32)
    return np.concatenate(features, axis=0)


def save_feature_pickle(features: np.ndarray, file_name: str) -> None:
    import pickle

    directory = os.path.dirname(file_name)
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(file_name, "wb") as handle:
        pickle.dump(features, handle)


def evaluate_zakynthos_predictions(
    df: pd.DataFrame,
    features_dir: str,
    method_prefix: str,
    flip: bool = True,
    grayscale: bool = False,
    mods: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Compare baseline vs LoRA MegaDescriptor on Zakynthos using Prediction."""
    from sides_matching.predictions import MegaDescriptor, Prediction

    if mods is None:
        mods = ["full", "same orientation", "different orientation"]

    path_query = os.path.join(
        features_dir,
        f"{method_prefix}_Zakynthos_flip={flip}_grayscale={grayscale}.pickle",
    )
    path_database = os.path.join(
        features_dir,
        f"{method_prefix}_Zakynthos_flip={False}_grayscale={grayscale}.pickle",
    )
    if not os.path.exists(path_query):
        raise FileNotFoundError(f"Missing query features: {path_query}")
    if not os.path.exists(path_database):
        raise FileNotFoundError(f"Missing database features: {path_database}")
    score_computer = MegaDescriptor(path_query, path_database)
    similarity = score_computer.compute_similarity(ignore="diagonal")
    prediction = Prediction(df, similarity, k=len(df) - 1)
    prediction.compute_accuracy(mods)
    rows = []
    for mod in mods:
        rows.append({
            "method": method_prefix,
            "flip": flip,
            "mod": mod,
            "top1": prediction.accuracy[mod]["top 1"],
            "top5": prediction.accuracy[mod]["top 5"],
        })
    return pd.DataFrame(rows)


def compare_base_vs_lora(
    df: pd.DataFrame,
    features_dir: str,
    flips: Sequence[bool] = (True, False),
    grayscale: bool = False,
    mods: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Run Zakynthos re-ID for base and LoRA and return long + side-by-side delta tables."""
    if mods is None:
        mods = [
            "full",
            "same orientation",
            "different orientation",
            "same year",
            "different year",
            "different both",
        ]
    results = []
    for flip in flips:
        for method_prefix in ("MegaDescriptor", "MegaDescriptorLoRA"):
            results.append(
                evaluate_zakynthos_predictions(
                    df,
                    features_dir,
                    method_prefix=method_prefix,
                    flip=flip,
                    grayscale=grayscale,
                    mods=list(mods),
                )
            )
    results_df = pd.concat(results, ignore_index=True)
    base = results_df[results_df["method"] == "MegaDescriptor"].rename(
        columns={"top1": "base_top1", "top5": "base_top5"}
    )
    lora = results_df[results_df["method"] == "MegaDescriptorLoRA"].rename(
        columns={"top1": "lora_top1", "top5": "lora_top5"}
    )
    comparison = base.merge(
        lora[["flip", "mod", "lora_top1", "lora_top5"]],
        on=["flip", "mod"],
        how="inner",
    )
    comparison["delta_top1"] = comparison["lora_top1"] - comparison["base_top1"]
    comparison["delta_top5"] = comparison["lora_top5"] - comparison["base_top5"]
    comparison = comparison[
        [
            "flip",
            "mod",
            "base_top1",
            "lora_top1",
            "delta_top1",
            "base_top5",
            "lora_top5",
            "delta_top5",
        ]
    ].sort_values(["flip", "mod"]).reset_index(drop=True)
    return results_df, comparison


def wildlife_dataset_from_df(
    root: str,
    df: pd.DataFrame,
    dataset_class: Callable,
    transform: Optional[T.Compose] = None,
) -> WildlifeDataset:
    return dataset_class(root, df=df, transform=transform)


def build_source_validation_loaders(
    train_sources: Sequence[Tuple[str, str, Callable]],
    val_df: pd.DataFrame,
    img_size: int,
    batch_size: int,
    num_workers: int,
) -> Tuple[Dict[str, DataLoader], Dict[str, np.ndarray], Dict[str, IdentityMapper]]:
    """Build per-source validation loaders with retrieval-only identity labels."""
    val_transform = get_eval_transform(flip=False, img_size=img_size)
    source_loaders: Dict[str, DataLoader] = {}
    source_orientations: Dict[str, np.ndarray] = {}
    source_mappers: Dict[str, IdentityMapper] = {}
    for name, root, dataset_fn in train_sources:
        source_val = val_df[val_df["source_dataset"] == name].reset_index(drop=True)
        if len(source_val) == 0:
            continue
        val_mapper = IdentityMapper.from_identities(source_val["global_identity"])
        source_mappers[name] = val_mapper
        val_dataset = wildlife_dataset_from_df(root, source_val, dataset_fn, val_transform)
        val_labels = val_mapper.encode_series(source_val["global_identity"])
        source_loaders[name] = build_concat_dataloader(
            [val_dataset],
            [val_labels],
            batch_size=max(batch_size * 2, 2),
            shuffle=False,
            num_workers=num_workers,
        )
        source_orientations[name] = source_val["orientation"].to_numpy()
    return source_loaders, source_orientations, source_mappers


def build_raw_train_datasets(
    train_sources: Sequence[Tuple[str, str, Callable]],
    train_df: pd.DataFrame,
    identity_mapper: IdentityMapper,
) -> Tuple[List[WildlifeDataset], List[np.ndarray], List[np.ndarray]]:
    """Build PIL-backed train datasets and aligned ArcFace / orientation arrays."""
    train_datasets: List[WildlifeDataset] = []
    train_label_parts: List[np.ndarray] = []
    train_orientation_parts: List[np.ndarray] = []
    for name, root, dataset_fn in train_sources:
        source_train = train_df[train_df["source_dataset"] == name].reset_index(drop=True)
        if len(source_train) == 0:
            continue
        train_datasets.append(wildlife_dataset_from_df(root, source_train, dataset_fn, transform=None))
        train_label_parts.append(identity_mapper.encode_series(source_train["global_identity"]))
        train_orientation_parts.append(source_train["orientation"].to_numpy())
    return train_datasets, train_label_parts, train_orientation_parts


def run_training_loop(
    model: MegaDescriptorLoRA,
    optimizer: torch.optim.Optimizer,
    train_loader: DataLoader,
    source_loaders: Dict[str, DataLoader],
    source_orientations: Dict[str, np.ndarray],
    identity_mapper: IdentityMapper,
    config: TrainConfig,
    device: torch.device,
    checkpoint_dir: str,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
    metrics_csv: Optional[str] = None,
) -> pd.DataFrame:
    """Train with source-only checkpoint selection and per-epoch audit trail."""
    best_metric = -1.0
    patience_counter = 0
    history_rows: List[Dict[str, object]] = []
    for epoch in range(1, config.epochs + 1):
        train_metrics = train_epoch(
            model,
            train_loader,
            optimizer,
            device,
            scaler=scaler,
            use_amp=config.use_amp,
            opposite_loss_weight=config.opposite_loss_weight,
            triplet_margin=config.triplet_margin,
            triplet_loss_weight=config.triplet_loss_weight,
        )
        val_metrics = evaluate_source_validation(
            model,
            source_loaders,
            source_orientations,
            device,
            use_amp=config.use_amp,
        )
        row: Dict[str, object] = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "loss_arc": train_metrics["loss_arc"],
            "loss_opp": train_metrics["loss_opp"],
            "loss_triplet": train_metrics["loss_triplet"],
            "macro_embed_recall": val_metrics["macro_embed_recall"],
            "macro_opp_recall": val_metrics["macro_opp_recall"],
        }
        for key, value in val_metrics["source_embed"].items():
            row[f"embed@{key}"] = value
        for key, value in val_metrics["source_opp"].items():
            row[f"opp@{key}"] = value
        history_rows.append(row)
        if metrics_csv:
            append_metrics_row(metrics_csv, row)
        checkpoint_score = float(val_metrics[config.checkpoint_metric])
        print(
            f"Epoch {epoch:02d} | train_loss={train_metrics['loss']:.4f} | "
            f"macro_opp={val_metrics['macro_opp_recall']:.4f} | "
            f"macro_embed={val_metrics['macro_embed_recall']:.4f}"
        )
        if checkpoint_score > best_metric:
            best_metric = checkpoint_score
            patience_counter = 0
            save_checkpoint(
                model,
                identity_mapper,
                config,
                checkpoint_dir,
                metrics={
                    config.checkpoint_metric: checkpoint_score,
                    "macro_embed_recall": float(val_metrics["macro_embed_recall"]),
                    "epoch": epoch,
                },
            )
            print(f"Saved best checkpoint to {checkpoint_dir}")
        else:
            patience_counter += 1
            if patience_counter >= config.early_stop_patience:
                print("Early stopping triggered (source macro opposite recall).")
                break
    return pd.DataFrame(history_rows)


def evaluate_holdout_opposite_recall(
    model: MegaDescriptorLoRA,
    holdout_df: pd.DataFrame,
    train_sources: Sequence[Tuple[str, str, Callable]],
    device: torch.device,
    config: TrainConfig,
) -> float:
    """Evaluate opposite-side recall on a held-out location pseudo-target."""
    if len(holdout_df) == 0:
        return 0.0
    val_transform = get_eval_transform(flip=False, img_size=config.img_size)
    holdout_mapper = IdentityMapper.from_identities(holdout_df["global_identity"])
    datasets: List[WildlifeDataset] = []
    label_parts: List[np.ndarray] = []
    orientation_parts: List[np.ndarray] = []
    for name, root, dataset_fn in train_sources:
        subset = holdout_df[holdout_df["source_dataset"] == name].reset_index(drop=True)
        if len(subset) == 0:
            continue
        datasets.append(wildlife_dataset_from_df(root, subset, dataset_fn, val_transform))
        label_parts.append(holdout_mapper.encode_series(subset["global_identity"]))
        orientation_parts.append(subset["orientation"].to_numpy())
    loader = build_concat_dataloader(
        datasets,
        label_parts,
        batch_size=max(config.batch_size * 2, 2),
        shuffle=False,
        num_workers=config.num_workers,
    )
    orientations = np.concatenate(orientation_parts)
    return opposite_embedding_recall_at_1(model, loader, orientations, device, use_amp=config.use_amp)


def run_ablation_smoke(
    recipe_name: str,
    config: TrainConfig,
    train_sources: Sequence[Tuple[str, str, Callable]],
    merged_df: pd.DataFrame,
    holdout_sources: Sequence[str],
    device: torch.device,
    checkpoint_root: str,
    epochs: int = 5,
    metrics_csv: Optional[str] = None,
) -> Dict[str, float]:
    """Short source-only ablation run for recipe comparison."""
    set_seed(config.seed)
    train_df, val_df, holdout_df = location_holdout_split(
        merged_df,
        holdout_sources=holdout_sources,
        val_fraction=config.val_identity_fraction,
        seed=config.seed,
    )
    identity_mapper = IdentityMapper.from_identities(train_df["global_identity"])
    train_datasets, train_label_parts, train_orientation_parts = build_raw_train_datasets(
        train_sources, train_df, identity_mapper
    )
    negative_map = build_negative_map_for_training(
        config, train_sources, train_df, identity_mapper, device
    )
    train_loader = build_train_loader(
        config, train_datasets, train_label_parts, train_orientation_parts, negative_map
    )
    source_loaders, source_orientations, _ = build_source_validation_loaders(
        train_sources,
        val_df,
        config.img_size,
        config.batch_size,
        config.num_workers,
    )
    model = build_training_model(identity_mapper.num_classes, config, device)
    optimizer = configure_optimizer(model, config)
    scaler = torch.cuda.amp.GradScaler(enabled=config.use_amp and device.type == "cuda")
    run_dir = os.path.join(checkpoint_root, f"{recipe_name}_seed{config.seed}")
    short_config = TrainConfig(**{**asdict(config), "epochs": epochs})
    history = run_training_loop(
        model,
        optimizer,
        train_loader,
        source_loaders,
        source_orientations,
        identity_mapper,
        short_config,
        device,
        run_dir,
        scaler=scaler,
        metrics_csv=metrics_csv,
    )
    holdout_opp = evaluate_holdout_opposite_recall(
        model, holdout_df, train_sources, device, config
    )
    if metrics_csv:
        append_metrics_row(
            metrics_csv,
            {
                "recipe": recipe_name,
                "seed": config.seed,
                "holdout_sources": ",".join(holdout_sources),
                "holdout_opp_recall": holdout_opp,
                "epochs_ran": len(history),
            },
        )
    del model, optimizer, scaler
    clear_cuda_memory()
    return {
        "holdout_opp_recall": holdout_opp,
        "macro_opp_recall": float(history["macro_opp_recall"].max()) if len(history) else 0.0,
    }


