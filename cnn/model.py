import torch
import torch.nn as nn
from torchvision import models


# WHY we freeze early layers:
# The early ResNet layers (conv1, layer1-3) learn low-level features like edges
# and textures that transfer well from ImageNet to CT patches.  Fine-tuning only
# layer4 + FC reduces overfitting on our small dataset (~500-2000 patches).


class ResNet18CACClassifier(nn.Module):

    def __init__(self, pretrained: bool = True, freeze_backbone: bool = True):
        super().__init__()

        # Load base ResNet-18 with optional ImageNet weights
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = models.resnet18(weights=weights)

        # Replace conv1: 3-channel RGB -> 1-channel greyscale CT
        old_conv = backbone.conv1
        new_conv = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

        if pretrained:
            # Average pretrained RGB weights across channel dim (64,3,7,7) -> (64,1,7,7)
            # Preserves learned spatial filters instead of random-init discarding them
            with torch.no_grad():
                new_conv.weight.copy_(old_conv.weight.mean(dim=1, keepdim=True))

        backbone.conv1 = new_conv

        # Replace final FC: 1000-class ImageNet head -> single binary logit
        # Output is a raw logit; use BCEWithLogitsLoss during training
        backbone.fc = nn.Linear(512, 1)

        self.model = backbone

        if freeze_backbone:
            # Freeze all parameters first ...
            for param in self.model.parameters():
                param.requires_grad = False

            # ... then unfreeze layer4 and FC so only they are trained
            for param in self.model.layer4.parameters():
                param.requires_grad = True
            for param in self.model.fc.parameters():
                param.requires_grad = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Standard ResNet forward pass; returns raw logits of shape (B, 1)
        return self.model(x)


# Instantiate and return a configured ResNet18CACClassifier
def get_model(pretrained: bool = True, freeze_backbone: bool = True) -> ResNet18CACClassifier:
    return ResNet18CACClassifier(pretrained=pretrained, freeze_backbone=freeze_backbone)


# Print total, trainable and frozen parameter counts for a given model
def count_trainable_params(model: nn.Module) -> None:
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters     : {total:,}")
    print(f"Trainable parameters : {trainable:,}")
    print(f"Frozen parameters    : {total - trainable:,}")


if __name__ == "__main__":
    model = get_model(pretrained=True, freeze_backbone=True)
    model.eval()

    dummy = torch.zeros(4, 1, 64, 64)  # batch of 4 greyscale 64x64 patches
    with torch.no_grad():
        out = model(dummy)

    print(f"Input  shape : {tuple(dummy.shape)}")
    print(f"Output shape : {tuple(out.shape)}")   # expected (4, 1)
    print()
    count_trainable_params(model)
