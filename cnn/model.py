import torch
import torch.nn as nn
from torchvision import models


# Freezing early layers prevents overfitting on our small medical dataset while preserving basic edge-detection knowledge.
# We only train the very last layer to specialize the network on coronary calcium.


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
            # Compresses the standard 3-channel color network into a 1-channel greyscale medical network.
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
        # Passes the medical image through the convolutional network to predict lesion probability.
        return self.model(x)


# A highly optimized EfficientNet architecture designed to consume minimal memory while maintaining high accuracy.
class EfficientNetB0CACClassifier(nn.Module):

    def __init__(self, pretrained: bool = True, freeze_backbone: bool = True):
        super().__init__()

        weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = models.efficientnet_b0(weights=weights)

        # Replace first conv: (32, 3, 3, 3) -> (32, 1, 3, 3) accepting greyscale
        old_conv = backbone.features[0][0]
        new_conv = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1, bias=False)

        if pretrained:
            # Compresses the standard 3-channel color network into a 1-channel greyscale medical network.
            with torch.no_grad():
                new_conv.weight.copy_(old_conv.weight.mean(dim=1, keepdim=True))

        backbone.features[0][0] = new_conv

        # Replace classifier: (1, 1280) ImageNet head -> single binary logit
        backbone.classifier = nn.Linear(1280, 1)

        self.model = backbone

        if freeze_backbone:
            # Freeze everything first …
            for param in self.model.parameters():
                param.requires_grad = False

            # … then unfreeze the last 3 MBConv blocks and the classifier
            for block in self.model.features[6:]:   # features[6], [7], [8] (conv head)
                for param in block.parameters():
                    param.requires_grad = True
            for param in self.model.classifier.parameters():
                param.requires_grad = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


# A lightweight 3-layer convolutional network built entirely from scratch without pretrained weights.
class CustomCNNCACClassifier(nn.Module):

    def __init__(self):
        super().__init__()

        def _block(in_ch, out_ch):
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            )

        self.features = nn.Sequential(
            _block(1,   32),   # 64 -> 32
            _block(32,  64),   # 32 -> 16
            _block(64,  128),  # 16 ->  8
            _block(128, 256),  # 8  ->  4
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))




_ARCHITECTURES = {"resnet18", "efficientnet", "custom"}


# Return a configured CAC classifier for the requested architecture.
# architecture: "resnet18" | "efficientnet" | "custom"
# pretrained:     load ImageNet weights (ignored for "custom")
# freeze_backbone: freeze early layers, fine-tune head only (ignored for "custom")
def get_model(
    architecture: str = "resnet18",
    pretrained: bool = True,
    freeze_backbone: bool = True,
) -> nn.Module:
    arch = architecture.lower()
    if arch == "resnet18":
        return ResNet18CACClassifier(pretrained=pretrained, freeze_backbone=freeze_backbone)
    if arch == "efficientnet":
        return EfficientNetB0CACClassifier(pretrained=pretrained, freeze_backbone=freeze_backbone)
    if arch == "custom":
        return CustomCNNCACClassifier()
    raise ValueError(
        f"Unknown architecture '{architecture}'. "
        f"Choose from: {sorted(_ARCHITECTURES)}"
    )


# Print total, trainable and frozen parameter counts for a given model
def count_trainable_params(model: nn.Module) -> None:
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters     : {total:,}")
    print(f"Trainable parameters : {trainable:,}")
    print(f"Frozen parameters    : {total - trainable:,}")


if __name__ == "__main__":
    dummy = torch.zeros(4, 1, 64, 64)  # batch of 4 greyscale 64x64 patches

    for arch in ("resnet18", "efficientnet", "custom"):
        model = get_model(architecture=arch, pretrained=True, freeze_backbone=True)
        model.eval()
        with torch.no_grad():
            out = model(dummy)
        print(f"\n[{arch}]")
        print(f"  Input  : {tuple(dummy.shape)}")
        print(f"  Output : {tuple(out.shape)}")   # expected (4, 1)
        count_trainable_params(model)
