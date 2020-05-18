import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau

from isr.models.lightning_model import LightningIsr


class SrCnn(LightningIsr):
    """
    Super resolution CNN
    https://arxiv.org/pdf/1501.00092.pdf
    """
    def __init__(self, hparams):
        super(SrCnn, self).__init__(hparams=hparams)
        self.model = nn.Sequential(
            nn.Conv2d(
                in_channels=self.hparams.num_channels,
                out_channels=self.hparams.layer_1_filters,
                kernel_size=self.hparams.layer_1_kernel,
                stride=1,
                padding=self.hparams.layer_1_kernel // 2
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=self.hparams.layer_1_filters,
                out_channels=self.hparams.layer_2_filters,
                kernel_size=self.hparams.layer_2_kernel,
                stride=1,
                padding=self.hparams.layer_2_kernel // 2
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=self.hparams.layer_2_filters,
                out_channels=self.hparams.num_channels,
                kernel_size=self.hparams.layer_3_kernel,
                stride=1,
                padding=self.hparams.layer_3_kernel // 2
            ),
            nn.ReLU(),
        )

    def forward(self, x):
        scaled = F.interpolate(
            x,
            scale_factor=self.hparams.scale_factor,
            align_corners=False,
            mode=self.hparams.upscale_mode
        )
        return torch.clamp(self.model(scaled), 0., 1.)

    def configure_optimizers(self):
        optim = torch.optim.SGD(
            self.parameters(),
            lr=self.hparams.learning_rate,
            momentum=self.hparams.momentum,
            weight_decay=self.hparams.weight_decay
        )
        scheduler = {
            'scheduler': ReduceLROnPlateau(optim, patience=20),
            'monitor': 'val_loss',
            'interval': 'epoch',
            'frequency': 1
        }
        return [optim], [scheduler]

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = LightningIsr.add_model_specific_args(parent_parser)
        parser.add_argument('--num_channels', type=int, default=3)
        parser.add_argument('--layer_1_filters', type=int, default=64)
        parser.add_argument('--layer_2_filters', type=int, default=32)
        parser.add_argument('--layer_1_kernel', type=int, default=9)
        parser.add_argument('--layer_2_kernel', type=int, default=1)
        parser.add_argument('--layer_3_kernel', type=int, default=5)
        parser.add_argument('--upscale_mode', type=str, default='bicubic')

        return parser
