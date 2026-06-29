import torch
import torch.nn as nn
import torchvision.models as models


class FreqEncoder(nn.Module):
    """
    Lightweight CNN encoder for the FFT log-magnitude spectrum.

    This module receives a precomputed single-channel frequency map
    [B, 1, H, W] and extracts compact features from it.

    The frequency map is computed in RRFreqDatasetFromCSV and passed
    to the model during training, so this encoder is a pure feature
    extractor — it does not compute the FFT itself.

    Why FFT for AI-generated image detection?
    AI-generated images (diffusion models, GANs) leave periodic artifacts
    in the frequency domain that are invisible to the human eye in RGB
    but show up clearly in the log-magnitude spectrum of the FFT.
    This encoder lets the backbone exploit those traces.
    """

    def __init__(self, out_features: int = 128):
        super().__init__()

        # Small convolutional network for 1-channel inputs.
        # Input shape: [B, 1, H, W]
        # Same architecture as SmallMapEncoder in model_depth.py
        # so that the two branches are comparable.
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

            # Collapse spatial dimensions into one global feature vector.
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        # Project the 64-dim vector to out_features.
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, out_features),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
        )

    def forward(self, x):
        """
        Args:
            x: Tensor [B, 1, H, W] — log-magnitude FFT spectrum.

        Returns:
            Tensor [B, out_features]
        """
        x = self.encoder(x)
        x = self.fc(x)
        return x


class FreqMultiTaskModel(nn.Module):
    """
    Multi-task model using:
        - RGB image processed by a pretrained ResNet18 backbone.
        - Log-magnitude FFT spectrum processed by FreqEncoder.

    The two feature vectors are fused and fed into two classification heads:
        1. fake_logits:  real vs AI-generated (binary)
        2. transform_logits: original vs internet-transmitted vs re-digitized (3-class)

    This follows the same design as GeometricMultiTaskModel in model_depth.py.
    """

    def __init__(
        self,
        num_transform_classes: int = 3,
        pretrained: bool = True,
        freq_features: int = 128,
        use_attention: bool = True,
    ):
        """
        Args:
            num_transform_classes:
                Number of transformation classes.
                    0 = original
                    1 = internet-transmitted
                    2 = re-digitized

            pretrained:
                If True, load ImageNet pretrained weights for the RGB backbone.

            freq_features:
                Size of the feature vector produced by FreqEncoder.

            use_attention:
                If True, apply a sigmoid gating module after fusion.
                This lets the model learn which fused features matter more.
        """
        super().__init__()

        self.use_attention = use_attention

        # RGB backbone: ResNet18 pretrained on ImageNet.
        # Same choice as model.py and model_depth.py for consistency.
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        rgb_backbone = models.resnet18(weights=weights)

        # Remove the original ImageNet classifier.
        # We only want the 512-dim feature vector.
        rgb_out_features = rgb_backbone.fc.in_features
        rgb_backbone.fc = nn.Identity()
        self.rgb_backbone = rgb_backbone

        # Frequency branch: extracts features from the FFT log-spectrum.
        self.freq_encoder = FreqEncoder(out_features=freq_features)

        # Total dimension after concatenating RGB and freq features.
        fusion_dim = rgb_out_features + freq_features

        # Linear fusion layer that projects the concatenated features
        # into a shared 512-dim representation.
        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
        )

        # Optional sigmoid gating module.
        # Learns to up-weight the most discriminative fused features.
        if self.use_attention:
            self.attention = nn.Sequential(
                nn.Linear(512, 512),
                nn.ReLU(inplace=True),
                nn.Linear(512, 512),
                nn.Sigmoid(),
            )
        else:
            self.attention = None

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

    def forward(self, images, freq_map):
        """
        Forward pass.

        Args:
            images:   RGB tensor [B, 3, H, W]
            freq_map: Log-magnitude FFT spectrum [B, 1, H, W]

        Returns:
            Dictionary with:
                fake_logits:       [B, 2]
                transform_logits:  [B, num_transform_classes]
        """

        # Extract RGB features using the pretrained backbone.
        rgb_features = self.rgb_backbone(images)

        # Extract frequency features from the FFT spectrum.
        freq_features = self.freq_encoder(freq_map)

        # Concatenate the two feature vectors.
        fused = torch.cat([rgb_features, freq_features], dim=1)

        # Project into a common shared representation.
        shared = self.fusion(fused)

        # Apply sigmoid gating if enabled.
        if self.use_attention:
            gate = self.attention(shared)
            shared = shared * gate

        # Task-specific predictions.
        fake_logits = self.fake_head(shared)
        transform_logits = self.transform_head(shared)

        return {
            "fake_logits": fake_logits,
            "transform_logits": transform_logits,
        }
