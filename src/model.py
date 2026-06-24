import torch.nn as nn
import torchvision.models as models


class RGBMultiTaskModel(nn.Module):
    """
    RGB baseline model for:
    - single-task real/fake classification
    - single-task transformation classification
    - joint multi-task classification   
    """

    def __init__(self, task: str = "multitask", num_transform_classes: int = 3, pretrained: bool = True):
        """
        Initialize the model.

        Args:
            task:
                "fake": single-task real/fake classification
                "transform": single-task transformation classification
                "multitask": joint multi-task classification 
            
            num_transform_classes:
                Number of transformation classes is 3:
                    0 = original
                    1 = internet-transmitted
                    2 = re-digitized

            pretrained:
                If True, use a ResNet18 pretrained on ImageNet.
        """

        super().__init__()

        if task not in ["fake", "transform", "multitask"]:
            raise ValueError(
                "task must be one of: 'fake', 'transform', 'multitask'"
            )

        self.task = task

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

                and/or

                transform_logits:
                    Raw scores for transformation classification.
                    Shape: [batch_size, 3]
        """

        # Extract shared visual features from the RGB images.
        features = self.backbone(images)

        # Output dictionary.
        outputs = {}

        # Add real/fake logits only if needed.
        if self.task in ["fake", "multitask"]:
            outputs["fake_logits"] = self.fake_head(features)

        # Add transformation logits only if needed.
        if self.task in ["transform", "multitask"]:
            outputs["transform_logits"] = self.transform_head(features)

        return outputs