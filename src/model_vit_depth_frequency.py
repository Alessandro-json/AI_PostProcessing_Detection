import torch
import torch.nn as nn
import timm


class SmallMapEncoder(nn.Module):
    """
    Lightweight CNN encoder for single-channel maps.

    Identical to SmallMapEncoder in model_depth_frequency.py.
    Used for both the depth branch and the frequency branch.

    Input:  [B, 1, H, W]
    Output: [B, out_features]
    """

    def __init__(self, out_features=128):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),

            nn.Linear(64, out_features),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
        )

    def forward(self, x):
        return self.encoder(x)


class ViTDepthFrequencyMultiTaskModel(nn.Module):
    """
    Multi-task model using RGB (ViT-Small), depth, and frequency information.

    Drop-in alternative to RGBDepthFrequencyMultiTaskModel
    (model_depth_frequency.py).

    The only difference is the RGB backbone:
        model_depth_frequency.py  →  ResNet18 (CNN, local features)
        this file                 →  ViT-Small (Transformer, global attention)

    The depth and frequency branches are unchanged: SmallMapEncoder (lightweight
    CNN) for each, since depth maps and frequency spectra are single-channel
    and ViT is not the right architecture for them.

    Why ViT for the RGB branch?
    AI-generated images can contain long-range periodic artifacts that
    self-attention captures better than local convolutions. Replacing only
    the RGB branch lets us isolate this effect while keeping the rest of
    the pipeline identical.

    Memory note:
    ViT-Small (~22M params) needs more GPU memory per sample than ResNet18
    (~11M params). Use batch_size=16 (not 32) on Colab free.
    """

    def __init__(
        self,
        num_transform_classes=3,
        pretrained=True,
        map_features=128,
        fusion_features=512,
        use_attention=True,
        vit_model_name="vit_small_patch16_224",
        freeze_backbone=False,
    ):
        """
        Args:
            num_transform_classes:
                Number of transformation classes (default 3):
                    0 = original
                    1 = internet-transmitted
                    2 = re-digitized

            pretrained:
                If True, load ImageNet-pretrained ViT weights via timm.

            map_features:
                Output dimension of each SmallMapEncoder (depth and frequency).
                Same default (128) as model_depth_frequency.py.

            fusion_features:
                Dimension of the shared fused representation.
                Same default (512) as model_depth_frequency.py.

            use_attention:
                If True, apply a sigmoid attention gate after fusion.
                Identical to model_depth_frequency.py.

            vit_model_name:
                timm model identifier for the RGB backbone.
                Default: "vit_small_patch16_224" (best for Colab free).
                Alternatives:
                    "vit_tiny_patch16_224"  — faster, less accurate
                    "vit_base_patch16_224"  — stronger, heavier (may OOM)

            freeze_backbone:
                If True, freeze all ViT weights and train only the
                map encoders, fusion block, and heads.
        """

        super().__init__()

        # --- RGB branch: ViT-Small backbone ---
        # num_classes=0 removes the built-in ImageNet head.
        # forward() returns the pooled CLS token: [B, rgb_features].
        self.rgb_backbone = timm.create_model(
            vit_model_name,
            pretrained=pretrained,
            num_classes=0,
        )

        # Feature dimension of the ViT output.
        #   vit_tiny_patch16_224  → 192
        #   vit_small_patch16_224 → 384  (default)
        #   vit_base_patch16_224  → 768
        rgb_features = self.rgb_backbone.num_features

        if freeze_backbone:
            for param in self.rgb_backbone.parameters():
                param.requires_grad = False

        # --- Depth branch ---
        # Identical to model_depth_frequency.py.
        self.depth_encoder = SmallMapEncoder(out_features=map_features)

        # --- Frequency branch ---
        # Identical to model_depth_frequency.py.
        self.frequency_encoder = SmallMapEncoder(out_features=map_features)

        # --- Fusion block ---
        # Concatenate RGB + depth + frequency features, then project.
        fusion_input_features = rgb_features + map_features + map_features

        self.fusion = nn.Sequential(
            nn.Linear(fusion_input_features, fusion_features),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
        )

        # --- Attention gate (optional) ---
        # Identical to model_depth_frequency.py.
        self.use_attention = use_attention

        if self.use_attention:
            self.attention_gate = nn.Sequential(
                nn.Linear(fusion_features, fusion_features),
                nn.ReLU(inplace=True),
                nn.Linear(fusion_features, fusion_features),
                nn.Sigmoid(),
            )

        # --- Task heads ---
        # Identical to model_depth_frequency.py.
        self.fake_head = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(fusion_features, 2),
        )

        self.transform_head = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(fusion_features, num_transform_classes),
        )

    def forward(self, images, depth, frequency):
        """
        Forward pass.

        Args:
            images:    RGB tensor    [B, 3, H, W]   — must be 224x224
            depth:     depth map     [B, 1, H, W]
            frequency: FFT spectrum  [B, 1, H, W]

        Returns:
            Dictionary with the same keys as RGBDepthFrequencyMultiTaskModel:
                fake_logits:      [B, 2]
                transform_logits: [B, num_transform_classes]
        """

        # RGB features from ViT backbone. Shape: [B, rgb_features]
        rgb_features = self.rgb_backbone(images)

        # Depth features from lightweight CNN. Shape: [B, map_features]
        depth_features = self.depth_encoder(depth)

        # Frequency features from lightweight CNN. Shape: [B, map_features]
        frequency_features = self.frequency_encoder(frequency)

        # Concatenate all three branches. Shape: [B, rgb_features + 2*map_features]
        fused_features = torch.cat(
            [rgb_features, depth_features, frequency_features],
            dim=1,
        )

        # Project to shared representation. Shape: [B, fusion_features]
        shared_features = self.fusion(fused_features)

        # Apply sigmoid attention gate if enabled.
        if self.use_attention:
            attention = self.attention_gate(shared_features)
            shared_features = shared_features * attention

        fake_logits      = self.fake_head(shared_features)
        transform_logits = self.transform_head(shared_features)

        return {
            "fake_logits":      fake_logits,
            "transform_logits": transform_logits,
        }
