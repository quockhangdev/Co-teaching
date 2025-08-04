# import torch
# import torch.nn as nn
# import torchvision.models as models


# class CNN(nn.Module):
#     def __init__(self, input_channel=3, n_outputs=12, pretrained=True):
#         super(CNN, self).__init__()

#         # Load DenseNet-121
#         self.backbone = models.densenet121(pretrained=pretrained)

#         # Modify first conv layer if input channel != 3
#         if input_channel != 3:
#             old_conv = self.backbone.features.conv0
#             self.backbone.features.conv0 = nn.Conv2d(
#                 input_channel,
#                 old_conv.out_channels,
#                 kernel_size=old_conv.kernel_size,
#                 stride=old_conv.stride,
#                 padding=old_conv.padding,
#                 bias=old_conv.bias is not None,
#             )

#         # Replace classifier
#         num_features = self.backbone.classifier.in_features
#         self.backbone.classifier = nn.Linear(num_features, n_outputs)

#     def forward(self, x):
#         return self.backbone(x)

import torch
import torch.nn as nn
import torchvision.models as models


class CNN(nn.Module):
    def __init__(self, input_channel=3, n_outputs=12, pretrained=True):
        super(CNN, self).__init__()

        # Load VGG16
        self.backbone = models.vgg16(pretrained=pretrained)

        # Modify first conv layer if input channel != 3
        if input_channel != 3:
            old_conv = self.backbone.features[0]
            new_conv = nn.Conv2d(
                input_channel,
                old_conv.out_channels,
                kernel_size=old_conv.kernel_size,
                stride=old_conv.stride,
                padding=old_conv.padding,
                bias=old_conv.bias is not None,
            )
            # Copy weights for existing channels
            if pretrained:
                with torch.no_grad():
                    if input_channel > 3:
                        new_conv.weight[:, :3, :, :] = old_conv.weight
                        # Initialize additional channels
                        nn.init.kaiming_normal_(
                            new_conv.weight[:, 3:, :, :],
                            mode="fan_out",
                            nonlinearity="relu",
                        )
                    else:
                        new_conv.weight[:, :input_channel, :, :] = old_conv.weight[
                            :, :input_channel, :, :
                        ]
            self.backbone.features[0] = new_conv

        # Replace classifier's last layer to match n_outputs
        # VGG16 classifier is a Sequential with last Linear at index 6
        num_features = self.backbone.classifier[6].in_features
        self.backbone.classifier[6] = nn.Linear(num_features, n_outputs)

    def forward(self, x):
        return self.backbone(x)
