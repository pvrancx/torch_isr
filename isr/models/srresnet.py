import torch
import torch.nn as nn

from isr.models.lightning_model import LightningIsr


class _ResidualBlock(nn.Module):
    def __init__(self, in_channels: int = 3, out_channels: int = 64):
        super(_ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.prelu = nn.PReLU(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        hid = self.conv1(x)
        hid = self.bn1(hid)
        hid = self.prelu(hid)
        hid = self.conv2(hid)
        out = self.bn2(hid)

        return torch.add(x, out)


class _SubPixelBlock(nn.Module):
    def __init__(self, in_channels: int = 64, out_channels: int = 64, scale_factor: int = 2):
        super(_SubPixelBlock, self).__init__()
        n_out = out_channels * scale_factor ** 2
        self.conv = nn.Conv2d(in_channels, n_out, kernel_size=3, stride=1, padding=1)
        self.shuffle = nn.PixelShuffle(scale_factor)
        self.prelu = nn.PReLU(out_channels)

    def forward(self, x):
        hid = self.conv(x)
        hid = self.shuffle(hid)
        out = self.prelu(hid)

        return out


class _ResNetModule(nn.Module):
    def __init__(self, in_channels: int = 64, out_channels: int = 64, n_blocks: int = 5):
        super(_ResNetModule, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=9, stride=4, padding=1)
        self.prelu = nn.PReLU(out_channels)

        self.resnet = nn.Sequential(*[_ResidualBlock(in_channels, out_channels)
                                      for _ in range(n_blocks)])

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        inp = self.prelu(self.conv1(x))

        hid = self.resnet(inp)

        out = self.bn(self.conv2(hid))

        return torch.add(inp, out)


class SrResNet(LightningIsr):
    def __init__(self, hparams):

        super(SrResNet, self).__init__(hparams)
        self.resnet = _ResNetModule(hparams.in_channels, hparams.hid_channels, hparams.n_blocks)
        self.scaler1 = _SubPixelBlock(hparams.hid_channels, hparams.hid_channels, 2)
        self.scaler2 = _SubPixelBlock(hparams.hid_channels, hparams.hid_channels, 2)
        self.conv = nn.Conv2d(
            hparams.hid_channels, hparams.in_channels, kernel_size=9, stride=1, padding=4
        )

    def forward(self, x):
        hid = self.resnet(x)
        upscale1 = self.scaler1(hid)
        upscale2 = self.scaler2(upscale1)
        out = self.conv(upscale2)
        return torch.clamp(out, 0., 1.)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = LightningIsr.add_model_specific_args(parent_parser)
        parser.add_argument('--in_channels', type=int, default=3)
        parser.add_argument('--hid_channels', type=int, default=64)
        parser.add_argument('--n_blocks', type=int, default=5)
        return parser

    def configure_optimizers(self):
        optim = torch.optim.Adam(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay
        )
        return optim







