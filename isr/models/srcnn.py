import torch
import torch.nn as nn
import torch.nn.functional as F


class SrCnn(nn.Module):
    """
    https://arxiv.org/pdf/1501.00092.pdf
    """
    def __init__(
            self,
            scale_factor: int = 2,
            num_channels: int = 3,
            layer_1_filters: int = 64,
            layer_2_filters: int = 32,
            layer_1_kernel: int = 9,
            layer_2_kernel: int = 1,
            layer_3_kernel: int = 5

    ):
        super(SrCnn, self).__init__()
        self.conv1 = nn.Conv2d(num_channels,
                               layer_1_filters,
                               kernel_size=layer_1_kernel,
                               padding=layer_1_kernel // 2
                               )
        self.conv2 = nn.Conv2d(layer_1_filters,
                               layer_2_filters,
                               kernel_size=layer_2_kernel,
                               padding=layer_2_kernel // 2
                               )
        self.conv3 = nn.Conv2d(layer_2_filters,
                               num_channels,
                               kernel_size=layer_3_kernel,
                               padding=layer_3_kernel // 2
                               )
        self.relu = nn.ReLU(inplace=True)
        self.scale_factor = scale_factor

    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.scale_factor, align_corners=False, mode='bicubic')
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        return torch.clamp(x, 0., 1.)
