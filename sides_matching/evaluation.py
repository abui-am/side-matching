"""Retrieval metrics, bootstrap statistics, and adapter verification."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F


def normalize_orientation_codes(orientations: Sequence) -> np.ndarray:
    """Map left/right style labels to {0, 1}; unknown -> -1."""
    codes = np.full(len(orientations), -1, dtype=np.int64)
    for idx, value in enumerate(orientations):
        text = str(value).strip().lower()
        if text in {"left", "l", "0"}:
            codes[idx] = 0
        elif text in {"right", "r", "1"}:
            codes[idx] = 1
    return codes


def identities_with_both_sides(
    df: pd.DataFrame,
    *,
    identity_col: str = "identity",
    orientation_col: str = "orientation",
) -> Set:
    """Return identities that have at least one left and one right photo."""
    codes = normalize_orientation_codes(df[orientation_col].to_numpy())
    identities = df[identity_col].to_numpy()
    bilateral: Set = set()
    for identity in pd.unique(identities):
        group_codes = codes[identities == identity]
        if (group_codes == 0).any() and (group_codes == 1).any():
            bilateral.add(identity)
    return bilateral


def filter_bilateral_df(
    df: pd.DataFrame,
    *,
    identity_col: str = "identity",
    orientation_col: str = "orientation",
) -> tuple[pd.DataFrame, np.ndarray]:
    """Keep rows whose identity has both left and right profile photos.

    Returns the filtered dataframe (reset index) and original row indices kept.
    """
    bilateral = identities_with_both_sides(
        df, identity_col=identity_col, orientation_col=orientation_col
    )
    keep_mask = df[identity_col].isin(bilateral).to_numpy()
    keep_idx = np.flatnonzero(keep_mask)
    filtered = df.iloc[keep_idx].reset_index(drop=True)
    return filtered, keep_idx


def subset_identities_df(
    df: pd.DataFrame,
    max_identities: int,
    seed: int,
    *,
    identity_col: str = "identity",
) -> pd.DataFrame:
    """Keep rows for up to ``max_identities`` identities (seeded random sample)."""
    if max_identities <= 0:
        raise ValueError("max_identities must be positive")
    identities = np.array(sorted(df[identity_col].unique()))
    if len(identities) <= max_identities:
        return df
    rng = np.random.default_rng(seed)
    chosen = sorted(rng.choice(identities, size=max_identities, replace=False))
    return df[df[identity_col].isin(chosen)].reset_index(drop=True)


def split_calibration_one_per_side(
    df: pd.DataFrame,
    seed: int,
    *,
    identity_col: str = "identity",
    orientation_col: str = "orientation",
) -> tuple[np.ndarray, np.ndarray]:
    """Pick one left + one right photo per identity for calibration; rest for test.

    Expects ``df`` to contain only bilateral identities (see ``filter_bilateral_df``).
    Unknown orientations (e.g. top views) are never selected for calibration.
    """
    codes = normalize_orientation_codes(df[orientation_col].to_numpy())
    identities = df[identity_col].to_numpy()
    rng = np.random.default_rng(seed)
    val: list[int] = []
    for identity in pd.unique(identities):
        positions = np.flatnonzero(identities == identity)
        group_codes = codes[positions]
        left_pool = positions[group_codes == 0]
        right_pool = positions[group_codes == 1]
        if left_pool.size == 0 or right_pool.size == 0:
            raise ValueError(f"Identity {identity!r} lacks left or right photos")
        val.append(int(rng.choice(left_pool)))
        val.append(int(rng.choice(right_pool)))
    val_arr = np.sort(np.asarray(val, dtype=np.int64))
    test_arr = np.setdiff1d(np.arange(len(df), dtype=np.int64), val_arr)
    if test_arr.size == 0:
        raise ValueError("Calibration split left no held-out test queries")
    return val_arr, test_arr


def _as_label_array(labels: Union[torch.Tensor, np.ndarray]) -> np.ndarray:
    if isinstance(labels, torch.Tensor):
        return labels.detach().cpu().numpy()
    return np.asarray(labels)


@dataclass
class RetrievalResult:
    recall_at_1: float
    recall_at_5: float
    mean_reciprocal_rank: float
    per_query_correct: np.ndarray
    per_query_ranks: np.ndarray


def embedding_recall_at_k(
    embeddings: torch.Tensor,
    labels: Union[torch.Tensor, np.ndarray],
    k: int = 1,
) -> float:
    """Cosine recall@k excluding self-matches."""
    labels_np = _as_label_array(labels)
    similarity = embeddings @ embeddings.T
    similarity.fill_diagonal_(-math.inf)
    preds = similarity.argsort(dim=1, descending=True)[:, :k]
    hits = np.array([
        labels_np[i] in labels_np[preds[i].tolist()]
        for i in range(len(labels_np))
    ], dtype=np.float32)
    return float(hits.mean()) if len(hits) else 0.0


def opposite_embedding_recall_at_k(
    embeddings: torch.Tensor,
    labels: Union[torch.Tensor, np.ndarray],
    orientations: Sequence,
    k: int = 1,
) -> float:
    """Top-k identity recall when gallery is restricted to the opposite side."""
    labels_np = _as_label_array(labels)
    ori = normalize_orientation_codes(orientations)
    if len(ori) != len(labels_np):
        raise ValueError("orientations must align with labels")
    similarity = embeddings @ embeddings.T
    similarity.fill_diagonal_(-math.inf)
    correct = 0
    total = 0
    for i in range(len(labels_np)):
        if ori[i] not in (0, 1):
            continue
        mask = ori == (1 - ori[i])
        if not bool(mask.any()):
            continue
        scores = similarity[i].clone()
        scores[~torch.as_tensor(mask)] = -math.inf
        topk = scores.argsort(descending=True)[:k]
        hit = any(labels_np[j] == labels_np[i] for j in topk.tolist())
        correct += int(hit)
        total += 1
    return correct / max(total, 1)


def compute_opposite_side_ranks(
    embeddings: torch.Tensor,
    labels: Union[torch.Tensor, np.ndarray],
    orientations: Sequence,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return per-query opposite-side rank (1-based) and top-1 correctness."""
    labels_np = _as_label_array(labels)
    ori = normalize_orientation_codes(orientations)
    similarity = embeddings @ embeddings.T
    similarity.fill_diagonal_(-math.inf)
    ranks = np.full(len(labels_np), np.nan, dtype=np.float64)
    correct = np.zeros(len(labels_np), dtype=np.float64)
    for i in range(len(labels_np)):
        if ori[i] not in (0, 1):
            continue
        mask = ori == (1 - ori[i])
        if not bool(mask.any()):
            continue
        scores = similarity[i].clone()
        scores[~torch.as_tensor(mask)] = -math.inf
        order = scores.argsort(descending=True)
        rank = 1
        for j in order.tolist():
            if labels_np[j] == labels_np[i]:
                ranks[i] = float(rank)
                correct[i] = float(rank == 1)
                break
            rank += 1
    return ranks, correct


def compute_retrieval_result(
    embeddings: torch.Tensor,
    labels: Union[torch.Tensor, np.ndarray],
    orientations: Optional[Sequence] = None,
    opposite_only: bool = False,
) -> RetrievalResult:
    if opposite_only:
        if orientations is None:
            raise ValueError("orientations required for opposite_only retrieval")
        ranks, correct = compute_opposite_side_ranks(embeddings, labels, orientations)
        valid = ~np.isnan(ranks)
        mrr = float(np.mean(1.0 / ranks[valid])) if valid.any() else 0.0
        recall1 = float(correct[valid].mean()) if valid.any() else 0.0
        recall5 = opposite_embedding_recall_at_k(embeddings, labels, orientations, k=5)
        return RetrievalResult(
            recall_at_1=recall1,
            recall_at_5=recall5,
            mean_reciprocal_rank=mrr,
            per_query_correct=correct,
            per_query_ranks=ranks,
        )
    recall1 = embedding_recall_at_k(embeddings, labels, k=1)
    recall5 = embedding_recall_at_k(embeddings, labels, k=5)
    labels_np = _as_label_array(labels)
    similarity = embeddings @ embeddings.T
    similarity.fill_diagonal_(-math.inf)
    order = similarity.argsort(dim=1, descending=True)
    ranks = np.full(len(labels_np), np.nan, dtype=np.float64)
    correct = np.zeros(len(labels_np), dtype=np.float64)
    for i in range(len(labels_np)):
        for rank, j in enumerate(order[i].tolist(), start=1):
            if labels_np[j] == labels_np[i]:
                ranks[i] = float(rank)
                correct[i] = float(rank == 1)
                break
    valid = ~np.isnan(ranks)
    mrr = float(np.mean(1.0 / ranks[valid])) if valid.any() else 0.0
    return RetrievalResult(
        recall_at_1=recall1,
        recall_at_5=recall5,
        mean_reciprocal_rank=mrr,
        per_query_correct=correct,
        per_query_ranks=ranks,
    )


@dataclass
class BootstrapResult:
    mean_delta: float
    ci_low: float
    ci_high: float
    n_bootstrap: int


def paired_identity_bootstrap(
    base_scores: np.ndarray,
    lora_scores: np.ndarray,
    identity_ids: np.ndarray,
    n_bootstrap: int = 2000,
    seed: int = 42,
    ci: float = 0.95,
) -> BootstrapResult:
    """Bootstrap paired deltas resampling whole identities."""
    if len(base_scores) != len(lora_scores) or len(base_scores) != len(identity_ids):
        raise ValueError("base_scores, lora_scores, and identity_ids must align")
    unique_ids = np.unique(identity_ids)
    id_to_indices: Dict[object, List[int]] = {}
    for idx, identity in enumerate(identity_ids):
        id_to_indices.setdefault(identity, []).append(idx)
    rng = np.random.default_rng(seed)
    deltas = []
    for _ in range(n_bootstrap):
        sampled_ids = rng.choice(unique_ids, size=len(unique_ids), replace=True)
        base_vals = []
        lora_vals = []
        for identity in sampled_ids:
            indices = id_to_indices[identity]
            base_vals.extend(base_scores[indices].tolist())
            lora_vals.extend(lora_scores[indices].tolist())
        deltas.append(float(np.mean(lora_vals) - np.mean(base_vals)))
    deltas_arr = np.asarray(deltas, dtype=np.float64)
    alpha = (1.0 - ci) / 2.0
    return BootstrapResult(
        mean_delta=float(np.mean(lora_scores - base_scores)),
        ci_low=float(np.quantile(deltas_arr, alpha)),
        ci_high=float(np.quantile(deltas_arr, 1.0 - alpha)),
        n_bootstrap=n_bootstrap,
    )


@torch.no_grad()
def verify_adapter_active(
    model: nn.Module,
    sample_input: torch.Tensor,
    atol: float = 1e-6,
) -> Tuple[bool, float]:
    """Return whether adapter-enabled and disabled outputs differ meaningfully."""
    if not hasattr(model, "disable_adapter_layers"):
        raise TypeError("model must be a PeftModel with adapter layers")
    model.eval()
    sample_input = sample_input.to(next(model.parameters()).device)
    with torch.no_grad():
        model.enable_adapter_layers()
        enabled = model(sample_input)
        model.disable_adapter_layers()
        disabled = model(sample_input)
        model.enable_adapter_layers()
    if not (torch.isfinite(enabled).all() and torch.isfinite(disabled).all()):
        return False, float("nan")
    max_diff = float((enabled - disabled).abs().max().item())
    return max_diff > atol, max_diff


@torch.no_grad()
def collect_embeddings(
    model: nn.Module,
    loader,
    device: torch.device,
    use_amp: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Collect L2-normalized embeddings and labels from a dataloader."""
    model.eval()
    embeddings: List[torch.Tensor] = []
    labels: List[torch.Tensor] = []
    amp_enabled = bool(use_amp and device.type == "cuda")
    encode = getattr(model, "encode", model)
    for batch in loader:
        if len(batch) == 2:
            images, batch_labels = batch
        else:
            images = batch[0]
            batch_labels = batch[1]
        images = images.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, enabled=amp_enabled):
            batch_embeddings = F.normalize(encode(images), dim=1)
        embeddings.append(batch_embeddings.float().cpu())
        labels.append(batch_labels if isinstance(batch_labels, torch.Tensor) else torch.as_tensor(batch_labels))
    if not embeddings:
        return torch.empty(0, 0), torch.empty(0, dtype=torch.long)
    return torch.cat(embeddings, dim=0), torch.cat(labels, dim=0)


def macro_average(values: Dict[str, float]) -> float:
    if not values:
        return 0.0
    return float(np.mean(list(values.values())))


def encode_identity_labels(identities: Sequence) -> np.ndarray:
    unique = {identity: idx for idx, identity in enumerate(sorted(set(identities)))}
    return np.array([unique[identity] for identity in identities], dtype=np.int64)


def opposite_top1_per_query(
    features: np.ndarray,
    identities: Sequence,
    orientations: Sequence,
) -> np.ndarray:
    """Per-query opposite-side top-1 correctness aligned with feature row order."""
    embeddings = torch.as_tensor(features, dtype=torch.float32)
    embeddings = F.normalize(embeddings, dim=1)
    labels = encode_identity_labels(identities)
    result = compute_retrieval_result(
        embeddings,
        labels,
        orientations=orientations,
        opposite_only=True,
    )
    valid = ~np.isnan(result.per_query_ranks)
    out = np.zeros(len(identities), dtype=np.float64)
    out[valid] = result.per_query_correct[valid]
    out[~valid] = np.nan
    return out


def compare_feature_opposite_bootstrap(
    base_features: np.ndarray,
    lora_features: np.ndarray,
    identities: Sequence,
    orientations: Sequence,
    n_bootstrap: int = 2000,
    seed: int = 42,
) -> BootstrapResult:
    base_scores = opposite_top1_per_query(base_features, identities, orientations)
    lora_scores = opposite_top1_per_query(lora_features, identities, orientations)
    valid = ~(np.isnan(base_scores) | np.isnan(lora_scores))
    return paired_identity_bootstrap(
        base_scores[valid],
        lora_scores[valid],
        np.asarray(identities)[valid],
        n_bootstrap=n_bootstrap,
        seed=seed,
    )
