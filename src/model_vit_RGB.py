import torch.nn as nn
import timm


class ViTRGBMultiTaskModel(nn.Module):
    """
    ViT-Small alternative to RGBMultiTaskModel.

    Drop-in replacement for RGBMultiTaskModel (model_RGB.py):
    the interface is identical — same constructor arguments, same
    forward() signature, same output dictionary keys — so train_vit_RGB.py
    can reuse the entire training loop from train_RGB.py without changes.

    The only difference is the backbone: Vision Transformer (ViT-Small,
    patch size 16, 224x224 input) instead of ResNet18.

    Why ViT as an alternative for AI-generated image detection?
    ResNet18 is a CNN: it captures local patterns via sliding convolutions.
    ViT captures global relationships via self-attention across 14x14=196
    non-overlapping patches. AI-generated images often contain long-range
    periodic artifacts that self-attention can detect more naturally than
    local convolutional filters.

    Memory note: ViT-Small (~22M params) uses more GPU memory per image
    than ResNet18 (~11M params). Use batch_size=16 on Colab free instead
    of the batch_size=32 used for the RGB baseline.
    """

    def __init__(
        self,
        task: str = "multitask",
        num_transform_classes: int = 3,
        pretrained: bool = True,
        vit_model_name: str = "vit_small_patch16_224",
        freeze_backbone: bool = False,
    ):
        """
        Args:
            task:
                "fake":      single-task real/fake classification.
                "transform": single-task transformation classification.
                "multitask": joint multi-task classification.
                (Same values as RGBMultiTaskModel.)

            num_transform_classes:
                Number of transformation classes (default 3):
                    0 = original
                    1 = internet-transmitted
                    2 = re-digitized

            pretrained:
                If True, load ImageNet-pretrained ViT weights via timm.
                Always True in practice; only set False for ablations.

            vit_model_name:
                timm model identifier. Default: "vit_small_patch16_224".
                Alternatives:
                    "vit_tiny_patch16_224"  — faster, less accurate
                    "vit_base_patch16_224"  — stronger, heavier (may OOM on Colab free)

            freeze_backbone:
                If True, only the two classification heads are trained.
                Useful as a quick sanity check or if GPU memory is tight.
        """

        super().__init__()

        if task not in ["fake", "transform", "multitask"]:
            raise ValueError(
                "task must be one of: 'fake', 'transform', 'multitask'"
            )

        self.task = task

        # Load ViT backbone from timm.
        # num_classes=0 removes the built-in ImageNet head so forward()
        # returns the raw pooled CLS token feature vector.
        self.backbone = timm.create_model(
            vit_model_name,
            pretrained=pretrained,
            num_classes=0,
        )

        # Feature dimension of the pooled output.
        #   vit_tiny_patch16_224  → 192
        #   vit_small_patch16_224 → 384   (default)
        #   vit_base_patch16_224  → 768
        in_features = self.backbone.num_features

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # Head 1: binary real/fake classification.
        # Same design as RGBMultiTaskModel.fake_head.
        self.fake_head = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(in_features, 2),
        )

        # Head 2: transformation classification.
        # Same design as RGBMultiTaskModel.transform_head.
        self.transform_head = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(in_features, num_transform_classes),
        )

    def forward(self, images):
        """
        Forward pass.

        Args:
            images: RGB tensor [B, 3, 224, 224].
                    Must be exactly 224x224 (divisible by patch size 16).

        Returns:
            A dictionary with the same keys as RGBMultiTaskModel.forward():

                fake_logits:      [B, 2]              (if task in fake / multitask)
                transform_logits: [B, num_transform_classes]  (if task in transform / multitask)
        """

        # Extract pooled feature vector from the ViT backbone.
        # Shape: [B, in_features]
        features = self.backbone(images)

        outputs = {}

        if self.task in ["fake", "multitask"]:
            outputs["fake_logits"] = self.fake_head(features)

        if self.task in ["transform", "multitask"]:
            outputs["transform_logits"] = self.transform_head(features)

        return outputs
