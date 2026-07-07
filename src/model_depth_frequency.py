import torch
import torch.nn as nn
from torchvision import models


class SmallMapEncoder(nn.Module):
    """
    Lightweight CNN encoder for single-channel maps.

    This encoder is used for:
    1. depth maps
    2. frequency maps
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


class RGBDepthFrequencyMultiTaskModel(nn.Module):
    """
    Multi-task model using RGB, depth, and frequency information.

    The extracted features are concatenated and passed through a fusion block.
    The fused representation is then used by two heads:
    1. real/fake classification head.
    2. transformation classification head.
    """

    def __init__(
        self,
        num_transform_classes=3,
        pretrained=True,
        map_features=128,
        fusion_features=512,
        use_attention=True,
    ):
        super().__init__()

        # Load the RGB backbone.
        try:
            weights = models.ResNet18_Weights.DEFAULT if pretrained else None
            self.rgb_backbone = models.resnet18(weights=weights)
        except AttributeError:
            self.rgb_backbone = models.resnet18(pretrained=pretrained)

        rgb_features = self.rgb_backbone.fc.in_features

        # The backbone will output RGB feature vectors.
        self.rgb_backbone.fc = nn.Identity()

        # Depth branch.
        self.depth_encoder = SmallMapEncoder(out_features=map_features)

        # Frequency branch.
        self.frequency_encoder = SmallMapEncoder(out_features=map_features)

        fusion_input_features = rgb_features + map_features + map_features

        # Shared fusion block.
        self.fusion = nn.Sequential(
            nn.Linear(fusion_input_features, fusion_features),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
        )

        self.use_attention = use_attention

        # Lightweight shared attention gate over fused features.
        # This allows the model to reweight fused features before the task heads.
        if self.use_attention:
            self.attention_gate = nn.Sequential(
                nn.Linear(fusion_features, fusion_features),
                nn.ReLU(inplace=True),
                nn.Linear(fusion_features, fusion_features),
                nn.Sigmoid(),
            )

        # Real/fake classification head.
        self.fake_head = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(fusion_features, 2),
        )

        # Transformation classification head.
        self.transform_head = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(fusion_features, num_transform_classes),
        )

    def forward(self, images, depth, frequency):
        """
        Forward pass
        """

        rgb_features = self.rgb_backbone(images)
        depth_features = self.depth_encoder(depth)
        frequency_features = self.frequency_encoder(frequency)

        fused_features = torch.cat(
            [rgb_features, depth_features, frequency_features],
            dim=1,
        )

        shared_features = self.fusion(fused_features)

        if self.use_attention:
            attention = self.attention_gate(shared_features)
            shared_features = shared_features * attention

        fake_logits = self.fake_head(shared_features)
        transform_logits = self.transform_head(shared_features)

        return {
            "fake_logits": fake_logits,
            "transform_logits": transform_logits,
        }