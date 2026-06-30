import torch
import torch.nn as nn
import timm


class ViTMultiTaskModel(nn.Module):
    """
    Multi-task model using a Vision Transformer (ViT-Small) backbone
    instead of the CNN backbone (ResNet18) used in model.py.

    Input: RGB image only (same as the baseline model.py).
    This lets us answer: does an attention-based architecture capture
    AI-generation artifacts better than a CNN, given the same input?

    Two independent classification heads sit on top of the shared
    ViT features, exactly like in model.py and model_freq.py:
        1. fake_logits:      real vs AI-generated (binary)
        2. transform_logits: original vs internet-transmitted vs re-digitized
    """

    def __init__(
        self,
        num_transform_classes: int = 3,
        pretrained: bool = True,
        vit_model_name: str = "vit_small_patch16_224",
        freeze_backbone: bool = False,
    ):
        """
        Args:
            num_transform_classes:
                Number of transformation classes (default 3).

            pretrained:
                If True, load ImageNet-pretrained ViT weights via timm.

            vit_model_name:
                Which timm ViT variant to use. Default is ViT-Small,
                the best balance of accuracy/speed for Colab free GPUs.
                Other options: "vit_tiny_patch16_224" (faster, weaker),
                "vit_base_patch16_224" (stronger, much heavier).

            freeze_backbone:
                If True, freezes all ViT weights and only trains the
                two classification heads. Useful if you run out of
                GPU memory or want faster epochs at the cost of accuracy.
        """
        super().__init__()

        # Load the ViT backbone from timm.
        # num_classes=0 removes the built-in classification head,
        # so forward() returns the raw feature vector (pooled CLS token).
        self.backbone = timm.create_model(
            vit_model_name,
            pretrained=pretrained,
            num_classes=0,
        )

        # Feature dimension produced by the backbone.
        # For vit_small_patch16_224 this is 384.
        # We read it dynamically so the code also works with vit_tiny (192)
        # or vit_base (768) without any change.
        backbone_out_features = self.backbone.num_features

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # Shared projection layer before the two task-specific heads.
        # Same design as model.py, keeps the two backbones comparable.
        self.shared = nn.Sequential(
            nn.Linear(backbone_out_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
        )

        # Head 1: binary real/fake classification.
        self.fake_head = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(512, 2),
        )

        # Head 2: transformation classification.
        self.transform_head = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(512, num_transform_classes),
        )

    def forward(self, images):
        """
        Forward pass.

        Args:
            images: RGB tensor [B, 3, 224, 224]
                    (must be exactly 224x224, divisible by the 16x16 patch size)

        Returns:
            Dictionary with:
                fake_logits:      [B, 2]
                transform_logits: [B, num_transform_classes]
        """

        # Extract the pooled feature vector from the ViT backbone.
        # Shape: [B, backbone_out_features]
        features = self.backbone(images)

        shared = self.shared(features)

        fake_logits = self.fake_head(shared)
        transform_logits = self.transform_head(shared)

        return {
            "fake_logits": fake_logits,
            "transform_logits": transform_logits,
        }
