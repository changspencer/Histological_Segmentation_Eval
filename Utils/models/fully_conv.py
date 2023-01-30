import torch
from torch import nn
import torch.nn.functional as F

class myFCN(nn.Module):
    """
    A fully-convolutional network with a feature extractor modeled after AlexNet.
    Original 'self.features' found at 
        https://github.com/pytorch/vision/blob/main/torchvision/models/alexnet.py.
    """
    def __init__(self, in_channels, num_classes, dropout=0.1):
        super().__init__()
        self.n_channels = in_channels
        self.n_classes = num_classes
        self.bilinear = True
        self.use_attention = False
        self.analyze = False

        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )  # Output size H / 4
        self.conv3 = nn.Sequential(
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )  # Output (H-1)/2
        self.conv4 = nn.Sequential(
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )  # Output (H-1)/2

        # Shortcut upsampling connections
        self.shortcut = nn.Sequential(
            nn.Conv2d(384, 96, 3, padding=1, bias=False),
            # nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Dropout(dropout),
            nn.Conv2d(96, num_classes, 1),
        )  # Output H x 2

        # Final Concatenation part - connects with output of self.conv4
        self.head = nn.Sequential(
            nn.Conv2d(256, 64, kernel_size=3, padding=1, bias=False),
            # nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False),
            nn.Dropout(dropout),
            nn.Conv2d(64, num_classes, 1),
        )
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

    def forward(self, x: torch.Tensor) -> torch.tensor:
        x2 = self.double_conv(x)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)

        # Shortcuts
        x3 = F.pad(x3, [0, 0, 0, 1])
        skip = self.shortcut(x3)

        # Consolidate shortcuts and FCN head output
        x4 = F.pad(x4, [0, 0, 0, 1])
        x5 = self.head(x4)
        x5 = x5[:, :, :skip.shape[-2], :skip.shape[-1]]  # Crop out extra pixels
        head = (x5 + skip) / 2  # Average the prediction values

        out = self.up(self.up(head))
        out = out[:, :, :x.shape[-2], :x.shape[-1]]  # Crop out extra pixels
        return out