"""Tests for retrieval evaluation helpers."""

import numpy as np
import torch

from sides_matching.evaluation import (
    compute_retrieval_result,
    embedding_recall_at_k,
    normalize_orientation_codes,
    opposite_embedding_recall_at_k,
    paired_identity_bootstrap,
    verify_adapter_active,
)


def test_normalize_orientation_codes():
    codes = normalize_orientation_codes(["left", "right", "L", "unknown"])
    assert codes.tolist() == [0, 1, 0, -1]


def test_perfect_embedding_recall():
    embeddings = torch.tensor([
        [1.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0],
        [0.0, 1.0],
    ])
    labels = np.array([0, 0, 1, 1])
    assert embedding_recall_at_k(embeddings, labels, k=1) == 1.0


def test_opposite_side_recall_perfect():
    embeddings = torch.tensor([
        [1.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0],
        [0.0, 1.0],
    ])
    labels = np.array([0, 0, 1, 1])
    orientations = ["left", "right", "left", "right"]
    assert opposite_embedding_recall_at_k(embeddings, labels, orientations, k=1) == 1.0


def test_bootstrap_is_deterministic():
    base = np.array([0.0, 1.0, 0.0, 1.0])
    lora = np.array([1.0, 1.0, 0.0, 0.0])
    identities = np.array(["a", "a", "b", "b"])
    first = paired_identity_bootstrap(base, lora, identities, n_bootstrap=100, seed=7)
    second = paired_identity_bootstrap(base, lora, identities, n_bootstrap=100, seed=7)
    assert first.mean_delta == second.mean_delta
    assert first.ci_low == second.ci_low
    assert first.ci_high == second.ci_high


def test_compute_retrieval_result_mrr():
    embeddings = torch.tensor([
        [1.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0],
        [0.0, 1.0],
    ])
    labels = np.array([0, 0, 1, 1])
    result = compute_retrieval_result(embeddings, labels, opposite_only=False)
    assert result.recall_at_1 == 1.0
    assert result.mean_reciprocal_rank == 1.0


class DummyPeftModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(4, 4)
        self.adapter_enabled = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.adapter_enabled:
            return self.linear(x) + 0.01
        return self.linear(x)

    def disable_adapter_layers(self) -> None:
        self.adapter_enabled = False

    def enable_adapter_layers(self) -> None:
        self.adapter_enabled = True


def test_verify_adapter_active():
    model = DummyPeftModel()
    sample = torch.randn(2, 4)
    active, diff = verify_adapter_active(model, sample)
    assert active is True
    assert diff > 0.0
