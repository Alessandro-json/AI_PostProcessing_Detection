# geometric_model.py

import torch
import torch.nn as nn
import torchvision.models as models


class SmallMapEncoder(nn.Module):
    """
    Lightweight CNN encoder for single-channel maps.

    Important:
    This module does NOT estimate depth from RGB images.
    It only receives precomputed depth maps and extracts compact features from them.

    We use this encoder for:
    - depth maps
    - edge-depth consistency maps
    """

    def __init__(self, out_features: int = 128):
        super().__init__()

        # Small convolutional network for 1-channel inputs.
        # Input shape: [B, 1, H, W]
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            # Convert spatial feature map into one global feature vector.
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        # Project the 64-dimensional feature vector to out_features.
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, out_features),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
        )

    def forward(self, x):
        """
        Args:
            x: Tensor with shape [B, 1, H, W]

        Returns:
            Tensor with shape [B, out_features]
        """
        x = self.encoder(x)
        x = self.fc(x)
        return x


class GeometricMultiTaskModel(nn.Module):
    """
    Multi-task model using:
    - RGB image
    - precomputed depth map
    - edge-depth consistency map

    The model still follows Project 2:
    one shared representation and two classification heads.

    Outputs:
    - fake_logits: real/fake prediction
    - transform_logits: transformation prediction
    """

    def __init__(
        self,
        num_transform_classes: int = 3,
        pretrained: bool = True,
        map_features: int = 128,
        use_edge: bool = True,
        use_attention: bool = True,
    ):
        super().__init__()

        self.use_edge = use_edge
        self.use_attention = use_attention

        # RGB backbone: same idea as the existing baseline.
        # We use ResNet18 pretrained on ImageNet.
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        rgb_backbone = models.resnet18(weights=weights)

        # Remove the original ImageNet classifier.
        rgb_features = rgb_backbone.fc.in_features
        rgb_backbone.fc = nn.Identity()

        self.rgb_backbone = rgb_backbone

        # Branch that extracts features from the precomputed depth map.
        self.depth_encoder = SmallMapEncoder(out_features=map_features)

        # Optional branch that extracts features from the edge-depth consistency map.
        if self.use_edge:
            self.edge_encoder = SmallMapEncoder(out_features=map_features)
            fusion_dim = rgb_features + map_features + map_features
        else:
            self.edge_encoder = None
            fusion_dim = rgb_features + map_features

        # Fuse RGB, depth, and optionally edge-consistency features.
        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
        )

        # Optional attention/gating module.
        # It learns which fused features are more useful.
        if self.use_attention:
            self.attention = nn.Sequential(
                nn.Linear(512, 512),
                nn.ReLU(inplace=True),
                nn.Linear(512, 512),
                nn.Sigmoid(),
            )
        else:
            self.attention = None

        # Head 1: binary classification, real vs fake.
        self.fake_head = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(512, 2),
        )

        # Head 2: transformation classification.
        # Example:
        # 0 = original
        # 1 = internet-transmitted
        # 2 = re-digitized
        self.transform_head = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(512, num_transform_classes),
        )

    def forward(self, images, depth, edge_consistency=None):
        """
        Args:
            images: RGB tensor [B, 3, H, W]
            depth: depth tensor [B, 1, H, W]
            edge_consistency: optional tensor [B, 1, H, W]

        Returns:
            Dictionary with:
            - fake_logits
            - transform_logits
        """

        # Extract RGB features.
        rgb_features = self.rgb_backbone(images)

        # Extract depth features.
        depth_features = self.depth_encoder(depth)

        # Collect all feature vectors before fusion.
        features = [rgb_features, depth_features]

        # Add edge-depth consistency features if enabled.
        if self.use_edge:
            if edge_consistency is None:
                raise ValueError(
                    "edge_consistency is required when use_edge=True."
                )

            edge_features = self.edge_encoder(edge_consistency)
            features.append(edge_features)

        # Concatenate all feature vectors.
        fused_features = torch.cat(features, dim=1)

        # Project them into a common shared representation.
        shared_features = self.fusion(fused_features)

        # Apply attention/gating if enabled.
        if self.use_attention:
            attention_gate = self.attention(shared_features)
            shared_features = shared_features * attention_gate

        # Task-specific predictions.
        fake_logits = self.fake_head(shared_features)
        transform_logits = self.transform_head(shared_features)

        return {
            "fake_logits": fake_logits,
            "transform_logits": transform_logits,
        }