import os
import sys
import csv
import argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # project root

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")  # headless backend — no display required
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
)

from cnn.model import get_model


# Load model weights from a checkpoint and set to eval mode
def _load_model(checkpoint_path: str, device: torch.device):
    model = get_model(pretrained=False, freeze_backbone=False)
    ckpt  = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()
    print(f"Loaded checkpoint: {checkpoint_path}  (saved at epoch {ckpt.get('epoch', '?')})")
    return model


# Run inference on every patch and return (true_labels, predicted_labels, confidence_scores)
def _infer_all(model, patches_np: np.ndarray, device: torch.device, batch_size: int = 64):
    all_probs = []
    n = len(patches_np)

    with torch.no_grad():
        for start in range(0, n, batch_size):
            batch = patches_np[start : start + batch_size]
            t = torch.from_numpy(batch).unsqueeze(1).to(device)  # (B, 1, 64, 64)
            logits = model(t)                                      # (B, 1)
            probs  = torch.sigmoid(logits).squeeze(1)             # (B,)
            all_probs.append(probs.cpu().numpy())

    return np.concatenate(all_probs)  # shape (N,)


# Derive a within-patient slice index by counting occurrences of each patient_id in order
def _make_slice_indices(patient_ids: np.ndarray) -> list:
    counters = {}
    indices  = []
    for pid in patient_ids:
        counters[pid] = counters.get(pid, -1) + 1
        indices.append(counters[pid])
    return indices


# Print a labelled 2x2 confusion matrix to stdout
def _print_confusion_matrix(cm: np.ndarray) -> None:
    tn, fp, fn, tp = cm.ravel()
    print("\nConfusion Matrix (rows = true, cols = predicted):")
    print(f"               Pred 0 (FP)   Pred 1 (CAC)")
    print(f"  True 0 (FP)    TN={tn:<6}    FP={fp}")
    print(f"  True 1 (CAC)   FN={fn:<6}    TP={tp}")


# Save a clean matplotlib confusion matrix figure
def _save_confusion_figure(cm: np.ndarray, out_path: str) -> None:
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.colorbar(im, ax=ax)

    labels = ["FP (0)", "CAC (1)"]
    ax.set_xticks([0, 1]); ax.set_xticklabels(labels)
    ax.set_yticks([0, 1]); ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title("Patch Confusion Matrix")

    thresh = cm.max() / 2.0
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black", fontsize=14)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Confusion matrix figure saved to: {out_path}")


def evaluate(
    npz_path: str,
    checkpoint_path: str = "cnn/checkpoints/best_model.pt",
    threshold: float = 0.5,
) -> None:
    os.makedirs("results", exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load data — no augmentation, all patches
    data        = np.load(npz_path, allow_pickle=True)
    patches     = data["patches"].astype(np.float32)      # (N, 64, 64)
    true_labels = data["labels"].astype(np.int32)         # (N,)
    patient_ids = data["patient_ids"]                     # (N,) object array of strings
    peak_hus    = data["peak_hus"].astype(np.float32)     # (N,)
    area_mm2s   = data["area_mm2s"].astype(np.float32)    # (N,)
    slice_idxs  = _make_slice_indices(patient_ids)        # derived per-patient index

    print(f"Evaluating on {len(patches)} patches from {npz_path} ...")

    model  = _load_model(checkpoint_path, device)
    probs  = _infer_all(model, patches, device)           # sigmoid confidences (N,)
    preds  = (probs >= threshold).astype(np.int32)        # binary predictions

    # Metrics
    acc  = accuracy_score(true_labels, preds)
    prec = precision_score(true_labels, preds, zero_division=0)
    rec  = recall_score(true_labels, preds, zero_division=0)
    f1   = f1_score(true_labels, preds, zero_division=0)
    auc  = roc_auc_score(true_labels, probs)
    cm   = confusion_matrix(true_labels, preds)

    print(f"\n--- Evaluation Metrics (threshold = {threshold}) ---")
    print(f"  Accuracy  : {acc  * 100:.2f}%")
    print(f"  Precision : {prec * 100:.2f}%")
    print(f"  Recall    : {rec  * 100:.2f}%")
    print(f"  F1 Score  : {f1  * 100:.2f}%")
    print(f"  ROC AUC   : {auc:.4f}")

    _print_confusion_matrix(cm)

    # Save confusion matrix figure
    _save_confusion_figure(cm, "results/patch_confusion_matrix.png")

    # Save per-patch predictions CSV
    csv_path = "results/patch_predictions.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["patient_id", "slice_idx", "area_mm2", "peak_hu",
                         "true_label", "predicted_label", "confidence"])
        for i in range(len(patches)):
            writer.writerow([
                patient_ids[i],
                slice_idxs[i],
                f"{area_mm2s[i]:.4f}",
                f"{peak_hus[i]:.1f}",
                int(true_labels[i]),
                int(preds[i]),
                f"{probs[i]:.4f}",
            ])
    print(f"Per-patch predictions saved to: {csv_path}")


# Plot training loss curves and val accuracy from the training log CSV
def plot_training_curves(log_csv: str = "cnn/checkpoints/training_log.csv") -> None:
    if not os.path.exists(log_csv):
        print(f"Training log not found: {log_csv}  — skipping curve plot.")
        return

    os.makedirs("results", exist_ok=True)

    epochs, train_losses, val_losses, val_accs = [], [], [], []
    with open(log_csv, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            epochs.append(int(row["epoch"]))
            train_losses.append(float(row["train_loss"]))
            val_losses.append(float(row["val_loss"]))
            val_accs.append(float(row["val_acc"]) * 100)  # convert to %

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))

    # Loss subplot
    ax1.plot(epochs, train_losses, label="Train loss", linewidth=1.8)
    ax1.plot(epochs, val_losses,   label="Val loss",   linewidth=1.8, linestyle="--")
    ax1.axvline(x=10.5, color="grey", linestyle=":", linewidth=1, label="Stage 2 start")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss")
    ax1.set_title("Training & Validation Loss")
    ax1.legend(); ax1.grid(alpha=0.3)

    # Accuracy subplot
    ax2.plot(epochs, val_accs, color="green", linewidth=1.8)
    ax2.axvline(x=10.5, color="grey", linestyle=":", linewidth=1, label="Stage 2 start")
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Val Accuracy (%)")
    ax2.set_title("Validation Accuracy")
    ax2.legend(); ax2.grid(alpha=0.3)

    plt.tight_layout()
    out_path = "results/training_curves.png"
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Training curves saved to: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate trained CAC classifier")
    parser.add_argument("--npz",   type=str, required=True,
                        help="Path to labelled patches .npz")
    parser.add_argument("--ckpt",  type=str, default="cnn/checkpoints/best_model.pt",
                        help="Path to model checkpoint")
    parser.add_argument("--thr",   type=float, default=0.5,
                        help="Sigmoid threshold for positive class (default 0.5)")
    parser.add_argument("--log",   type=str, default="cnn/checkpoints/training_log.csv",
                        help="Path to training log CSV for curve plot")
    args = parser.parse_args()

    evaluate(args.npz, args.ckpt, args.thr)
    plot_training_curves(args.log)
