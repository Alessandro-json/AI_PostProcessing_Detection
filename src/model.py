import torch.nn as nn
import torchvision.models as models


class RGBMultiTaskModel(nn.Module):
    """
    RGB multi-task baseline model.

    The model receives one RGB image and produces two predictions:

        1. fake_logits:
            real vs AI-generated image

        2. transform_logits:
            original vs internet-transmitted vs re-digitized image

    The architecture is composed by a shared backbone with two classification heads trained jointly.
    """

    def __init__(self, num_transform_classes: int = 3, pretrained: bool = True):
        """
        Initialize the model.

        Args:
            num_transform_classes:
                Number of transformation classes is 3:
                    0 = original
                    1 = internet-transmitted
                    2 = re-digitized

            pretrained:
                If True, use a ResNet18 pretrained on ImageNet.
        """

        super().__init__()

        # Select the ResNet18 weights.
        # If pretrained=True, we load ImageNet pretrained weights.
        # If pretrained=False, the model starts with random weights.
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None

        # Load a ResNet18 model from torchvision.
        backbone = models.resnet18(weights=weights)

        # The original ResNet18 ends with a fully connected layer that predicts ImageNet classes.
        # But we only want the feature vector so we remove the original classifier.
        in_features = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.backbone = backbone

        # First classification head:
        # predicts whether the image is real or AI-generated.
        self.fake_head = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(in_features, 2),
        )

        # Second classification head:
        # predicts which post-processing transformation was applied.
        self.transform_head = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(in_features, num_transform_classes),
        )

    def forward(self, images):
        """
        Forward pass of the model.

        Returns:
            A dictionary containing:

                fake_logits:
                    Raw scores for real/fake classification.
                    Shape: [batch_size, 2]

                transform_logits:
                    Raw scores for transformation classification.
                    Shape: [batch_size, 3]
        """

        # Extract shared visual features from the RGB images.
        features = self.backbone(images)

        # Use the shared features for the first task:
        # real vs fake classification.
        fake_logits = self.fake_head(features)

        # Use the same shared features for the second task:
        # transformation classification.
        transform_logits = self.transform_head(features)

        # Return both predictions.
        return {
            "fake_logits": fake_logits,
            "transform_logits": transform_logits,
        }