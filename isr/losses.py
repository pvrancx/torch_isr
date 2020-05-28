import torch
import torch.nn as nn
from torchvision import models


def pixel_loss(x: torch.Tensor, y: torch.Tensor):
    return nn.functional.mse_loss(x, y)


class FeatureLoss(nn.Module):
    def __init__(self, layer_idx: int = 8):
        super(FeatureLoss, self).__init__()
        assert 0 <= layer_idx <= 36, 'Layer index must be in [1, 36]'
        self.features = models.vgg19(pretrained=True).features[:layer_idx+1]

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        return nn.functional.mse_loss(self.features(x), self.features(y))
