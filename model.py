import torch
import torch.nn as nn
import torchvision.models as models


class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class CNN(nn.Module):
    def __init__(self, input_channel=3, n_outputs=12, pretrained=True, add_se=True):
        super(CNN, self).__init__()
        self.backbone = models.densenet121(pretrained=pretrained)

        # Modify input conv if needed
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

        # Add SE blocks if requested
        if add_se:
            self._insert_se_blocks()

        # Replace classifier with init
        num_features = self.backbone.classifier.in_features
        self.backbone.classifier = nn.Linear(num_features, n_outputs)
        self._init_classifier(self.backbone.classifier)

    def _insert_se_blocks(self):
        se_channels = {
            "denseblock1": 256,
            "denseblock2": 512,
            "denseblock3": 1024,
            "denseblock4": 1024  # trước transition4 (nếu dùng) sẽ giảm nữa
        }
        for name in se_channels:
            block = getattr(self.backbone.features, name)
            se = SEBlock(se_channels[name])
            wrapped = nn.Sequential(block, se)
            setattr(self.backbone.features, name, wrapped)

    def _init_classifier(self, layer):
        nn.init.kaiming_normal_(layer.weight, mode="fan_out", nonlinearity="relu")
        if layer.bias is not None:
            nn.init.constant_(layer.bias, 0)

    def forward(self, x):
        return self.backbone(x)
