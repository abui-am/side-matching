"""LoRA fine-tuning helpers for MegaDescriptor with ArcFace metric learning."""

from __future__ import annotations

import json
import math
import os
from dataclasses import asdict, dataclass
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from peft import LoraConfig, PeftModel, get_peft_model
from torch.utils.data import DataLoader, Dataset
from wildlife_datasets.datasets import WildlifeDataset

MODEL_NAME = "hf-hub:BVRA/MegaDescriptor-L-384"
DEFAULT_LORA_TARGETS = ("qkv", "proj", "fc1", "fc2")


@dataclass
class TrainConfig:
    model_name: str = MODEL_NAME
    img_size: int = 384
    # MegaDescriptor-L @ 384 needs small batches on Colab T4 (15GB).
    batch_size: int = 2
    epochs: int = 20
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_target_modules: Tuple[str, ...] = DEFAULT_LORA_TARGETS
    lr_lora: float = 1e-4
    lr_head: float = 1e-3
    weight_decay: float = 0.01
    arcface_s: float = 30.0
    arcface_m: float = 0.3
    val_identity_fraction: float = 0.2
    early_stop_patience: int = 5
    seed: int = 42
    use_amp: bool = True
    grad_checkpointing: bool = True
    num_workers: int = 2


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


def get_train_transform(img_size: int = 384) -> T.Compose:
    return T.Compose([
        T.Resize([img_size, img_size]),
        T.RandomHorizontalFlip(),
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.02),
        T.ToTensor(),
        T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])


def get_eval_transform(flip: bool = False, img_size: int = 384) -> T.Compose:
    transforms: List[nn.Module] = [T.Resize([img_size, img_size])]
    if flip:
        transforms.append(T.RandomHorizontalFlip(1))
    transforms.extend([
        T.ToTensor(),
        T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])
    return T.Compose(transforms)


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
        # nn.Module is callable; never invoke it to "unwrap" or Swin.forward misses `x`.
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


def trainable_parameters(model: MegaDescriptorLoRA) -> List[nn.Parameter]:
    params = [p for p in model.backbone.parameters() if p.requires_grad]
    params.extend(model.arcface_head.parameters())
    return params


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


def train_epoch(
    model: MegaDescriptorLoRA,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
    use_amp: bool = True,
) -> float:
    model.train()
    total_loss = 0.0
    n_batches = 0
    amp_enabled = bool(use_amp and device.type == "cuda")
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=amp_enabled):
            logits = model(images, labels)
            loss = F.cross_entropy(logits, labels)
        if scaler is not None and amp_enabled:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        total_loss += float(loss.detach().item())
        n_batches += 1
    return total_loss / max(n_batches, 1)


@torch.no_grad()
def validate_epoch(
    model: MegaDescriptorLoRA,
    loader: DataLoader,
    device: torch.device,
    use_amp: bool = True,
) -> Tuple[float, float]:
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    amp_enabled = bool(use_amp and device.type == "cuda")
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, enabled=amp_enabled):
            logits = model(images, labels)
            loss = F.cross_entropy(logits, labels)
        total_loss += float(loss.item())
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    avg_loss = total_loss / max(len(loader), 1)
    recall_at_1 = correct / max(total, 1)
    return avg_loss, recall_at_1


@torch.no_grad()
def embedding_recall_at_1(
    model: MegaDescriptorLoRA,
    loader: DataLoader,
    device: torch.device,
    use_amp: bool = True,
) -> float:
    """Cosine recall@1 on the validation split (same protocol as re-ID)."""
    model.eval()
    embeddings = []
    labels = []
    amp_enabled = bool(use_amp and device.type == "cuda")
    for images, batch_labels in loader:
        images = images.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, enabled=amp_enabled):
            batch_embeddings = F.normalize(model.encode(images), dim=1)
        embeddings.append(batch_embeddings.float().cpu())
        labels.append(batch_labels)
    if not embeddings:
        return 0.0
    embeddings = torch.cat(embeddings, dim=0)
    labels = torch.cat(labels, dim=0)
    similarity = embeddings @ embeddings.T
    similarity.fill_diagonal_(-math.inf)
    preds = similarity.argmax(dim=1)
    return (labels[preds] == labels).float().mean().item()


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
        return np.empty((0, model.num_features), dtype=np.float32)
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
    # dataset wrappers (amvrakikos, etc.) already set img_load='auto'
    return dataset_class(root, df=df, transform=transform)
