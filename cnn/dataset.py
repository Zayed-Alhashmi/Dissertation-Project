import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # project root

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import train_test_split


# Injects random optical distortions like rotations and brightness changes to artificially expand training data.
def _augment(patch: np.ndarray) -> np.ndarray:
    if np.random.rand() < 0.5:
        patch = np.fliplr(patch)
    if np.random.rand() < 0.5:
        patch = np.flipud(patch)

    angle = np.random.uniform(-15, 15)
    from scipy.ndimage import rotate
    patch = rotate(patch, angle, reshape=False, order=1, mode="constant", cval=0.0)

    brightness = np.random.uniform(0.85, 1.15)
    patch = np.clip(patch * brightness, 0.0, 1.0)

    return patch.astype(np.float32)


class PatchDataset(Dataset):
    # Loads patches and labels from a .npz file and optionally applies augmentation at retrieval time.
    def __init__(self, npz_path: str, augment: bool = False):
        data = np.load(npz_path, allow_pickle=True)
        self.patches = data["patches"].astype(np.float32)  # shape (N, 64, 64)
        self.labels  = data["labels"].astype(np.int64)     # shape (N,)
        self.augment = augment

    def __len__(self) -> int:
        return len(self.labels)

    # Returns a single (tensor, label) pair with the patch shaped as (1, H, W) for a single-channel input.
    def __getitem__(self, idx: int):
        patch = self.patches[idx].copy()

        if self.augment:
            patch = _augment(patch)

        tensor = torch.from_numpy(patch).unsqueeze(0)  # add channel dim → (1, 64, 64)
        label  = int(self.labels[idx])
        return tensor, label


# Divides the total data into a learning set and a testing set, balancing the rarity of actual calcium lesions.
def get_dataloaders(
    npz_path: str,
    batch_size: int = 32,
    val_split: float = 0.2,
    seed: int = 42,
):
    full_dataset = PatchDataset(npz_path, augment=False)
    labels = full_dataset.labels

    indices = np.arange(len(full_dataset))
    train_idx, val_idx = train_test_split(
        indices,
        test_size=val_split,
        stratify=labels,
        random_state=seed,
    )

    train_dataset = PatchDataset(npz_path, augment=True)   # augmentation on for training
    val_dataset   = PatchDataset(npz_path, augment=False)  # no augmentation for validation

    train_loader = DataLoader(
        Subset(train_dataset, train_idx),
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
    )
    val_loader = DataLoader(
        Subset(val_dataset, val_idx),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    n_pos = int((labels == 1).sum())
    n_neg = int((labels == 0).sum())
    total = len(labels)
    weight_neg = total / (2.0 * n_neg) if n_neg > 0 else 1.0  # higher weight for the minority class
    weight_pos = total / (2.0 * n_pos) if n_pos > 0 else 1.0
    class_weights = torch.tensor([weight_neg, weight_pos], dtype=torch.float32)

    print(f"  Train: {len(train_idx)} samples  |  Val: {len(val_idx)} samples")
    print(f"  Class weights  - neg (0): {weight_neg:.3f}  pos (1): {weight_pos:.3f}")

    return train_loader, val_loader, class_weights


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m cnn.dataset <labelled_patches.npz>")
        sys.exit(1)

    npz_path = sys.argv[1]

    train_loader, val_loader, class_weights = get_dataloaders(npz_path)

    batch_tensors, batch_labels = next(iter(train_loader))
    print(f"\nFirst batch shape : {batch_tensors.shape}")
    print(f"Label values      : {batch_labels.tolist()[:10]}")
    print(f"Class weights     : {class_weights}")

    full = PatchDataset(npz_path)
    n_pos = int((full.labels == 1).sum())
    n_neg = int((full.labels == 0).sum())
    print(f"\nFull dataset - CAC (1): {n_pos}  |  FP (0): {n_neg}  |  Total: {len(full)}")
