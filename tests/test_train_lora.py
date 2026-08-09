"""Tests for LoRA training helpers."""

import numpy as np
import pandas as pd
import torch
from PIL import Image

from sides_matching.train_lora import (
    AUGMENT_POLICIES,
    HardNegativeMap,
    IdentityMapper,
    OppositeTripletDataset,
    PairedAugmentTransform,
    _triplet_cosine_loss,
    merge_train_dataframes,
    split_identities,
    split_identities_stratified,
)


class DummyWildlifeDataset:
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        return self.images[index]


def _solid_image(color: tuple[int, int, int]) -> Image.Image:
    array = np.zeros((32, 32, 3), dtype=np.uint8)
    array[:, :] = color
    return Image.fromarray(array)


def test_split_identities_is_disjoint():
    df = pd.DataFrame({
        "global_identity": ["a", "a", "b", "b", "c", "c"],
        "source_dataset": ["Amvrakikos"] * 6,
    })
    train_df, val_df = split_identities(df, seed=42)
    train_ids = set(train_df["global_identity"])
    val_ids = set(val_df["global_identity"])
    assert train_ids.isdisjoint(val_ids)


def test_split_identities_stratified_per_source():
    df = pd.DataFrame({
        "global_identity": [f"a{i}" for i in range(4)] + [f"b{i}" for i in range(4)],
        "source_dataset": ["Amvrakikos"] * 4 + ["ReunionGreen"] * 4,
    })
    train_df, val_df = split_identities_stratified(df, val_fraction=0.5, seed=7)
    for source in df["source_dataset"].unique():
        source_train = set(train_df[train_df["source_dataset"] == source]["global_identity"])
        source_val = set(val_df[val_df["source_dataset"] == source]["global_identity"])
        assert source_train.isdisjoint(source_val)
        assert len(source_train) > 0
        assert len(source_val) > 0


def test_identity_mapper_train_only_excludes_val():
    train_df = pd.DataFrame({"global_identity": ["a", "b", "c"]})
    val_df = pd.DataFrame({"global_identity": ["d", "e"]})
    mapper = IdentityMapper.from_identities(train_df["global_identity"])
    assert mapper.num_classes == 3
    assert "d" not in mapper.identity_to_label


def test_paired_augment_shared_blur_kernel():
    policy = AUGMENT_POLICIES["standard"]
    transform = PairedAugmentTransform(policy, img_size=32)
    rng = np.random.default_rng(0)
    anchor = _solid_image((255, 0, 0))
    positive = _solid_image((0, 255, 0))
    first_a, first_b = transform.apply_pair(anchor, positive, rng)
    rng = np.random.default_rng(0)
    second_a, second_b = transform.apply_pair(anchor, positive, rng)
    assert torch.allclose(first_a, second_a)
    assert torch.allclose(first_b, second_b)


def test_minimal_policy_has_no_blur_or_flip():
    policy = AUGMENT_POLICIES["minimal"]
    assert policy.blur_p == 0.0
    assert policy.flip_p == 0.0


def test_hard_negative_map_uses_different_label():
    embeddings = np.array([
        [1.0, 0.0],
        [0.9, 0.1],
        [0.0, 1.0],
    ], dtype=np.float32)
    labels = np.array([0, 0, 1], dtype=np.int64)
    negative_map = HardNegativeMap(embeddings, labels, np.random.default_rng(0), top_k=3)
    assert labels[negative_map.get(0)] != labels[0]
    assert labels[negative_map.get(2)] != labels[2]


def test_triplet_loss_margin():
    anchor = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
    positive = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    negative = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
    loss = _triplet_cosine_loss(anchor, positive, negative, margin=0.2)
    assert loss.item() >= 0.0


def test_opposite_triplet_dataset_labels():
    images = [_solid_image((i, 0, 0)) for i in range(4)]
    dataset_a = DummyWildlifeDataset(images, [0, 0, 1, 1])
    labels = np.array([0, 0, 1, 1], dtype=np.int64)
    orientations = [np.array(["left", "right", "left", "right"])]
    embeddings = np.eye(4, dtype=np.float32)
    negative_map = HardNegativeMap(embeddings, labels, np.random.default_rng(0), top_k=4)
    paired = PairedAugmentTransform(AUGMENT_POLICIES["minimal"], img_size=32)
    triplet = OppositeTripletDataset(
        [dataset_a],
        [labels],
        orientations,
        negative_map=negative_map,
        paired_augment=paired,
        seed=1,
    )
    anchor, positive, negative, anchor_label, negative_label = triplet[0]
    assert anchor_label.item() != negative_label.item()
    assert anchor.shape == positive.shape == negative.shape
