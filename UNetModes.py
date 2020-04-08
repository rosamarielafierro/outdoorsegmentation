import torch
from torch import nn
import torch.nn.functional as F

class UpConv(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

       
        if bilinear:
            self.up = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding= 1))
        else:
            self.up = nn.ConvTranspose2d(in_channels , out_channels, kernel_size=2, stride=2)

    def forward(self, x):
        return self.up(x)


class UNetFull(nn.Module):
    def __init__(self, num_classes,  input_filters = 3, start_features= 64, bilinear=True):
        super().__init__()

        self.num_classes = num_classes
        self.input_filters = input_filters
        self.start_features = start_features

        self.down_layers1 = nn.Sequential(
          nn.Conv2d(self.input_filters, self.start_features, 3, padding = 1),
          nn.BatchNorm2d(self.start_features),
          nn.ReLU(),
          nn.Conv2d(self.start_features, self.start_features, 3, padding = 1),
          nn.BatchNorm2d(self.start_features),
          nn.ReLU()
        )

        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.down_layers2 = nn.Sequential(
          nn.Conv2d(self.start_features, self.start_features * 2, 3, padding = 1),
          nn.BatchNorm2d(self.start_features * 2),
          nn.ReLU(),
          nn.Conv2d(self.start_features * 2, self.start_features * 2, 3, padding = 1),
          nn.BatchNorm2d(self.start_features * 2),
          nn.ReLU()
        )

        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.down_layers3 = nn.Sequential(
          nn.Conv2d(self.start_features * 2, self.start_features * 4, 3, padding = 1),
          nn.BatchNorm2d(self.start_features * 4),
          nn.ReLU(),
          nn.Conv2d(self.start_features * 4, self.start_features * 4, 3, padding = 1),
          nn.BatchNorm2d(self.start_features * 4),
          nn.ReLU()
        )

        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.down_layers4 = nn.Sequential(
          nn.Conv2d(self.start_features * 4, self.start_features * 8, 3, padding = 1),
          nn.BatchNorm2d(self.start_features * 8),
          nn.ReLU(),
          nn.Conv2d(self.start_features * 8, self.start_features * 8, 3, padding = 1),
          nn.BatchNorm2d(self.start_features * 8),
          nn.ReLU()
        )

        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.base_layer = nn.Sequential(
          nn.Conv2d(self.start_features * 8, self.start_features * 16, 3, padding = 1),
          nn.BatchNorm2d(self.start_features * 16),
          nn.ReLU(),
          nn.Conv2d(self.start_features * 16, self.start_features * 16, 3, padding = 1),
          nn.BatchNorm2d(self.start_features * 16),
          nn.ReLU()
        )

        self.up_conv4 = UpConv(self.start_features * 16, self.start_features *8, bilinear=bilinear)

        self.up_layers4 = nn.Sequential(
          nn.Conv2d(self.start_features * 16, self.start_features * 8, 3, padding = 1),
          nn.BatchNorm2d(self.start_features * 8),
          nn.ReLU(),
          nn.Conv2d(self.start_features * 8, self.start_features * 8, 3, padding = 1),
          nn.BatchNorm2d(self.start_features * 8),
          nn.ReLU()
        )
        self.up_conv3 = UpConv(self.start_features * 8, self.start_features *4, bilinear=bilinear)

        self.up_layers3 = nn.Sequential(
          nn.Conv2d(self.start_features * 8, self.start_features * 4, 3, padding = 1),
          nn.BatchNorm2d(self.start_features * 4),
          nn.ReLU(),
          nn.Conv2d(self.start_features * 4, self.start_features * 4, 3, padding = 1),
          nn.BatchNorm2d(self.start_features * 4),
          nn.ReLU()
        )

        self.up_conv2 = UpConv(self.start_features * 4, self.start_features *2, bilinear=bilinear)

        self.up_layers2 = nn.Sequential(
          nn.Conv2d(self.start_features * 4, self.start_features * 2, 3, padding = 1),
          nn.BatchNorm2d(self.start_features * 2),
          nn.ReLU(),
          nn.Conv2d(self.start_features * 2, self.start_features * 2, 3, padding = 1),
          nn.BatchNorm2d(self.start_features * 2),
          nn.ReLU()
        )
        self.up_conv1 = UpConv(self.start_features * 2, self.start_features, bilinear=bilinear)

        self.up_layers1 = nn.Sequential(
          nn.Conv2d(self.start_features * 2, self.start_features, 3, padding = 1),
          nn.BatchNorm2d(self.start_features),
          nn.ReLU(),
          nn.Conv2d(self.start_features, self.start_features, 3, padding = 1),
          nn.BatchNorm2d(self.start_features),
          nn.ReLU(),
        )

        self.conv = nn.Conv2d(self.start_features, num_classes, kernel_size=1)

    def forward(self, x):

        down1 = self.down_layers1(x)
        down2 = self.down_layers2(self.pool1(down1))
        down3 = self.down_layers3(self.pool2(down2))
        down4 = self.down_layers4(self.pool3(down3))

        x = self.base_layer(self.pool4(down4))

        x = self.up_conv4(x)
        x = torch.cat([x, down4], dim=1)
        x = self.up_layers4(x)

        x = self.up_conv3(x)
        x = torch.cat([x, down3], dim=1)
        x = self.up_layers3(x)

        x = self.up_conv2(x)
        x = torch.cat([x, down2], dim=1)
        x = self.up_layers2(x)

        x = self.up_conv1(x)
        x = torch.cat([x, down1], dim=1)
        x = self.up_layers1(x)

        x = self.conv(x)

        return x