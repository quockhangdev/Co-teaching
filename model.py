import torch
import torch.nn as nn
import torchvision.models as models


class CNN(nn.Module):
    def __init__(self, input_channel=3, n_outputs=12, pretrained=True):
        super(CNN, self).__init__()

        # Load DenseNet-121
        self.backbone = models.densenet121(pretrained=pretrained)

        # Modify first conv layer if input channel != 3
        if input_channel != 3:
            old_conv = self.backbone.features.conv0
            self.backbone.features.conv0 = nn.Conv2d(
                input_channel,
                old_conv.out_channels,
                kernel_size=old_conv.kernel_size,
                stride=old_conv.stride,
                padding=old_conv.padding,
                bias=old_conv.bias is not None,
            )

        # Replace classifier
        num_features = self.backbone.classifier.in_features
        self.backbone.classifier = nn.Linear(num_features, n_outputs)

    def forward(self, x):
        return self.backbone(x)
