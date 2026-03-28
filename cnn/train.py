import os
import sys
import csv
import argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # project root

import torch
import torch.nn as nn

from cnn.model import get_model
from cnn.dataset import get_dataloaders


# Two-stage fine-tuning strategy:
#   Stage 1 (epochs 1-10)  — backbone frozen, only layer4 + FC trained (fast convergence, low overfit risk)
#   Stage 2 (epoch 11+)    — full backbone unfrozen, lr reduced 10x (slow, careful global refinement)
def _unfreeze_all(model: nn.Module, optimizer: torch.optim.Optimizer, new_lr: float) -> None:
    for param in model.model.parameters():
        param.requires_grad = True
    for group in optimizer.param_groups:  # update every param group in-place
        group["lr"] = new_lr
    print(f"  [Stage 2] Full backbone unfrozen — lr reduced to {new_lr:.2e}")


# Run one training epoch and return the average loss
def _train_epoch(model, loader, criterion, optimizer, device) -> float:
    model.train()
    total_loss = 0.0

    for patches, labels in loader:
        patches = patches.to(device)
        labels  = labels.float().unsqueeze(1).to(device)  # (B,) -> (B, 1) for BCEWithLogitsLoss

        optimizer.zero_grad()
        logits = model(patches)         # (B, 1) raw logit
        loss   = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * patches.size(0)  # accumulate weighted by batch size

    return total_loss / len(loader.dataset)


# Run one validation epoch; return (avg_loss, accuracy)
def _val_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct    = 0

    with torch.no_grad():
        for patches, labels in loader:
            patches = patches.to(device)
            labels  = labels.float().unsqueeze(1).to(device)

            logits = model(patches)
            loss   = criterion(logits, labels)
            total_loss += loss.item() * patches.size(0)

            preds    = (logits > 0).long()   # threshold raw logit at 0 → class 1
            correct += (preds == labels.long()).sum().item()

    avg_loss = total_loss / len(loader.dataset)
    accuracy = correct / len(loader.dataset)
    return avg_loss, accuracy


def train(
    npz_path: str,
    epochs: int = 25,
    lr: float = 1e-4,
    batch_size: int = 32,
    checkpoint_dir: str = "cnn/checkpoints/",
) -> None:
    os.makedirs(checkpoint_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Data
    print("\nBuilding dataloaders ...")
    train_loader, val_loader, class_weights = get_dataloaders(npz_path, batch_size)

    # Model
    model = get_model(pretrained=True, freeze_backbone=True)
    model = model.to(device)

    # Loss — positive class upweighted to handle class imbalance
    pos_weight = class_weights[1].to(device)
    criterion  = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # Optimiser and scheduler
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
        weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=5, factor=0.5
    )

    # CSV log
    log_path = os.path.join(checkpoint_dir, "training_log.csv")
    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "val_loss", "val_acc"])

    best_val_loss  = float("inf")
    best_val_acc   = 0.0
    best_epoch     = 0
    stage2_entered = False

    print(f"\nStarting training — {epochs} epochs\n")

    for epoch in range(1, epochs + 1):

        # Stage 2: unfreeze full backbone after epoch 10
        if epoch == 11 and not stage2_entered:
            _unfreeze_all(model, optimizer, new_lr=lr / 10)
            # Rebuild optimiser so all parameters are included
            optimizer = torch.optim.Adam(
                model.parameters(), lr=lr / 10, weight_decay=1e-4
            )
            stage2_entered = True

        train_loss          = _train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc   = _val_epoch(model, val_loader, criterion, device)

        scheduler.step(val_loss)

        print(
            f"Epoch {epoch:>3}/{epochs} | "
            f"Train loss: {train_loss:.3f} | "
            f"Val loss: {val_loss:.3f} | "
            f"Val acc: {val_acc * 100:.1f}%"
        )

        # Checkpoint: save whenever val loss improves
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc  = val_acc
            best_epoch    = epoch
            ckpt_path = os.path.join(checkpoint_dir, "best_model.pt")
            torch.save({"epoch": epoch, "model_state_dict": model.state_dict(),
                        "val_loss": val_loss, "val_acc": val_acc}, ckpt_path)
            print(f"  -> Checkpoint saved (val loss improved)")

        # Append to CSV log every epoch
        with open(log_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([epoch, f"{train_loss:.5f}", f"{val_loss:.5f}", f"{val_acc:.5f}"])

    print(f"\nTraining complete.")
    print(f"Best val accuracy: {best_val_acc * 100:.1f}%  at epoch {best_epoch}")
    print(f"Checkpoint saved to : {os.path.join(checkpoint_dir, 'best_model.pt')}")
    print(f"Training log saved to: {log_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ResNet18 CAC classifier")
    parser.add_argument("--npz",        type=str,   required=True,               help="Path to labelled patches .npz")
    parser.add_argument("--epochs",     type=int,   default=25,                  help="Number of epochs (default 25)")
    parser.add_argument("--lr",         type=float, default=1e-4,                help="Initial learning rate (default 1e-4)")
    parser.add_argument("--batch_size", type=int,   default=32,                  help="Batch size (default 32)")
    parser.add_argument("--ckpt_dir",   type=str,   default="cnn/checkpoints/",  help="Checkpoint output directory")
    args = parser.parse_args()

    train(
        npz_path       = args.npz,
        epochs         = args.epochs,
        lr             = args.lr,
        batch_size     = args.batch_size,
        checkpoint_dir = args.ckpt_dir,
    )
