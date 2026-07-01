import torch
import torch.nn as nn
from torchvision import models


class SmallMapEncoder(nn.Module):
    """
    Lightweight CNN encoder for single-channel maps such as depth maps
    and frequency maps.
    """

    def __init__(self, output_dim=128):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.projection = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, output_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.projection(x)
        return x


class RGBDepthFrequencyMultiTaskModel(nn.Module):
    """
    RGB + Depth + Frequency multi-task model with task-specific gated fusion.

    Instead of simply concatenating all features and using the same fused
    representation for both tasks, this model learns two different modality
    gates:

    1. one gate for the real/fake task
    2. one gate for the transformation task

    This allows the model to learn whether RGB, depth, or frequency is more
    useful for each specific task.
    """

    def __init__(
        self,
        num_fake_classes=2,
        num_transform_classes=3,
        pretrained=True,
        use_attention=True,
        shared_dim=512,
        map_feature_dim=128,
        dropout=0.3,
    ):
        super().__init__()

        # RGB branch: ResNet18 backbone.
        try:
            weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
            self.rgb_backbone = models.resnet18(weights=weights)
        except AttributeError:
            self.rgb_backbone = models.resnet18(pretrained=pretrained)

        rgb_feature_dim = self.rgb_backbone.fc.in_features
        self.rgb_backbone.fc = nn.Identity()

        # Depth and frequency branches.
        self.depth_encoder = SmallMapEncoder(output_dim=map_feature_dim)
        self.frequency_encoder = SmallMapEncoder(output_dim=map_feature_dim)

        # Project all modalities to the same feature dimension.
        self.rgb_projector = nn.Sequential(
            nn.Linear(rgb_feature_dim, shared_dim),
            nn.LayerNorm(shared_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

        self.depth_projector = nn.Sequential(
            nn.Linear(map_feature_dim, shared_dim),
            nn.LayerNorm(shared_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

        self.frequency_projector = nn.Sequential(
            nn.Linear(map_feature_dim, shared_dim),
            nn.LayerNorm(shared_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

        # Gate input is the concatenation of the three projected modalities.
        gate_input_dim = shared_dim * 3

        # Task-specific modality gates.
        # Each gate outputs 3 values: weight_RGB, weight_depth, weight_frequency.
        self.fake_gate = nn.Sequential(
            nn.Linear(gate_input_dim, shared_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(shared_dim, 3),
        )

        self.transform_gate = nn.Sequential(
            nn.Linear(gate_input_dim, shared_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(shared_dim, 3),
        )

        self.use_attention = use_attention

        if self.use_attention:
            self.fake_attention = nn.Sequential(
                nn.Linear(shared_dim, shared_dim),
                nn.ReLU(inplace=True),
                nn.Linear(shared_dim, shared_dim),
                nn.Sigmoid(),
            )

            self.transform_attention = nn.Sequential(
                nn.Linear(shared_dim, shared_dim),
                nn.ReLU(inplace=True),
                nn.Linear(shared_dim, shared_dim),
                nn.Sigmoid(),
            )

        # Task-specific heads.
        self.fake_head = nn.Sequential(
            nn.Linear(shared_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_fake_classes),
        )

        self.transform_head = nn.Sequential(
            nn.Linear(shared_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_transform_classes),
        )

    def forward(self, images, depth, frequency):
        # Extract modality-specific features.
        rgb_features = self.rgb_backbone(images)
        depth_features = self.depth_encoder(depth)
        frequency_features = self.frequency_encoder(frequency)

        # Project all modalities to the same dimension.
        rgb_features = self.rgb_projector(rgb_features)
        depth_features = self.depth_projector(depth_features)
        frequency_features = self.frequency_projector(frequency_features)

        # Shape: [batch_size, 3, shared_dim]
        modality_features = torch.stack(
            [rgb_features, depth_features, frequency_features],
            dim=1,
        )

        # Shape: [batch_size, shared_dim * 3]
        gate_input = torch.cat(
            [rgb_features, depth_features, frequency_features],
            dim=1,
        )

        # Task-specific modality weights.
        fake_weights = torch.softmax(self.fake_gate(gate_input), dim=1)
        transform_weights = torch.softmax(self.transform_gate(gate_input), dim=1)

        # Weighted sum of modalities for each task.
        fake_fused = torch.sum(
            modality_features * fake_weights.unsqueeze(-1),
            dim=1,
        )

        transform_fused = torch.sum(
            modality_features * transform_weights.unsqueeze(-1),
            dim=1,
        )

        # Optional task-specific feature attention.
        if self.use_attention:
            fake_fused = fake_fused * self.fake_attention(fake_fused)
            transform_fused = transform_fused * self.transform_attention(transform_fused)

        fake_logits = self.fake_head(fake_fused)
        transform_logits = self.transform_head(transform_fused)

        return {
            "fake_logits": fake_logits,
            "transform_logits": transform_logits,

            # These are useful for analysis, but not used for the loss.
            "fake_modality_weights": fake_weights,
            "transform_modality_weights": transform_weights,
        }