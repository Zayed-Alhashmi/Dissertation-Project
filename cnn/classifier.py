import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # project root

import numpy as np
import torch

from cnn.model import get_model


class CACClassifier:
    # Load the model once at construction - not per-slice - to keep inference fast
    def __init__(
        self,
        checkpoint_path: str | None = None,
        arch: str = "resnet18",
        patch_size: int = 64,
        threshold: float = 0.5,
    ):
        self.patch_size = patch_size
        self.threshold  = threshold
        self.arch       = arch   # stored so callers can verify which model is loaded
        self.device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Resolve default checkpoint path from arch when none is provided
        if checkpoint_path is None:
            checkpoint_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "cnn", "checkpoints", f"best_model_{arch}.pt"
            )

        model = get_model(architecture=arch, pretrained=False, freeze_backbone=False)
        ckpt  = torch.load(checkpoint_path, map_location=self.device)
        model.load_state_dict(ckpt["model_state_dict"])
        model.to(self.device).eval()
        self.model            = model
        self.checkpoint_path  = checkpoint_path   # stored for logging

    # Extract a patch_size x patch_size crop centred on (cy, cx), zero-padded at borders
    def _extract_patch(self, hu_slice: np.ndarray, cy: float, cx: float) -> np.ndarray:
        h, w   = hu_slice.shape
        half   = self.patch_size // 2
        r0, r1 = int(cy) - half, int(cy) - half + self.patch_size
        c0, c1 = int(cx) - half, int(cx) - half + self.patch_size

        # compute valid overlap with the image
        sr0 = max(r0, 0); sr1 = min(r1, h)
        sc0 = max(c0, 0); sc1 = min(c1, w)

        patch = np.zeros((self.patch_size, self.patch_size), dtype=np.float32)
        pr0 = sr0 - r0; pr1 = pr0 + (sr1 - sr0)
        pc0 = sc0 - c0; pc1 = pc0 + (sc1 - sc0)
        patch[pr0:pr1, pc0:pc1] = hu_slice[sr0:sr1, sc0:sc1]
        return patch

    # Filter a list of regionprops blobs for a single CT slice.
    # Returns only the blobs that the CNN classifies as true CAC (class 1).
    # All blobs from the slice are batched into one forward pass for speed.
    def filter_blobs(self, hu_slice: np.ndarray, blobs: list, spacing) -> list:
        if not blobs:  # nothing to classify, return immediately
            return []

        patches = []
        for blob in blobs:
            cy, cx = blob.centroid  # regionprops centroid is (row, col)
            patch  = self._extract_patch(hu_slice, cy, cx)

            # Normalise: clip to [0, 1000] HU then scale to [0, 1]
            patch  = np.clip(patch, 0.0, 1000.0) / 1000.0
            patches.append(patch)

        # Stack into (B, 1, 64, 64) tensor - single batched forward pass
        batch = torch.from_numpy(np.stack(patches, axis=0)).unsqueeze(1).to(self.device)

        with torch.no_grad():
            logits = self.model(batch)                     # (B, 1)
            probs  = torch.sigmoid(logits).squeeze(1)      # (B,)
            keep   = (probs >= self.threshold).cpu().numpy()  # boolean mask (B,)

        return [blob for blob, flag in zip(blobs, keep) if flag]
