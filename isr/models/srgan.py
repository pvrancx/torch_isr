from argparse import ArgumentParser
from collections import OrderedDict
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from pytorch_lightning import LightningModule
from torch.autograd import Variable

from isr.models import SrResNet
from isr.models.lightning_model import LightningIsr


class _DiscriminatorBlock(nn.Module):
    def __init__(
            self, in_filters: int = 3, out_filters: int = 64, kernel_size: int = 3, stride: int = 1
    ):
        super(_DiscriminatorBlock, self).__init__()
        self.conv = nn.Conv2d(in_filters, out_filters, kernel_size=kernel_size, stride=stride)
        self.bn = nn.BatchNorm2d(num_features=out_filters)
        self.lrelu = nn.LeakyReLU(inplace=True)

    def forward(self, x):
        return self.lrelu(self.bn(self.conv(x)))


class Discriminator(LightningModule):
    def __init__(self, hparams):
        super(Discriminator, self).__init__()
        self.hparams = hparams
        self.loss = nn.BCELoss()

        self.conv = nn.Conv2d(hparams.in_channels, 64, kernel_size=3, stride=1)
        self.lrelu = nn.LeakyReLU()

        self.model = nn.Sequential(
            _DiscriminatorBlock(64, 64, 3, 2),
            _DiscriminatorBlock(64, 128, 3, 1),
            _DiscriminatorBlock(128, 128, 3, 2),
            _DiscriminatorBlock(128, 256, 3, 1),
            _DiscriminatorBlock(256, 256, 3, 2),
            _DiscriminatorBlock(256, 512, 3, 1),
            _DiscriminatorBlock(512, 512, 3, 2)
        )

        n_features = self._get_n_features((hparams.in_channels,) + hparams.img_shape)

        self.classifier = nn.Sequential(
            nn.Linear(n_features, 1024),
            nn.LeakyReLU(),
            nn.Linear(1024, 1),
            nn.Sigmoid()
        )

    def _forward_features(self, x):
        inp = self.lrelu(self.conv(x))
        features = self.model(inp)
        return features

    def _get_n_features(self, input_shape: Tuple[int, int, int]):
        inp = Variable(torch.rand(1, *input_shape))
        output_feat = self._forward_features(inp)
        return output_feat.data.view(1, -1).size(1)

    def forward(self, x):
        inp = self.lrelu(self.conv(x))
        features = self.model(inp)
        out = self.classifier(features.view(x.size(0), -1))

        return out

    def training_step(self, batch, batch_nb):
        x, y = batch
        loss = self.loss(self(x), y)
        tensorboard_logs = {'train_loss': loss}
        return {'loss': loss, 'log': tensorboard_logs}

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        logits = self.forward(x)
        loss = self.loss(logits, y)
        return {'val_loss': loss}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        tensorboard_logs = {'val_loss': avg_loss}
        return {'avg_val_loss': avg_loss, 'log': tensorboard_logs}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--lr', type=float, default=0.002, help='base learning rate')
        parser.add_argument('--in_channels', type=int, default=3)
        parser.add_argument('--img_shape', type=tuple, default=(28, 28))

        return parser


class SrGan(LightningIsr):

    def __init__(self, hparams):
        super(SrGan, self).__init__(hparams)
        self.hparams = hparams

        self.generator = SrResNet(hparams)
        self.discriminator = Discriminator(hparams)

        self.generated_imgs = None
        self.last_imgs = None

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = Discriminator.add_model_specific_args(parent_parser)
        parser = SrResNet.add_model_specific_args(parser)

        return parser

    def forward(self, z):
        return self.generator(z)

    def adversarial_loss(self, y_hat, y):
        return F.binary_cross_entropy(y_hat, y)

    def training_step(self, batch, batch_nb, optimizer_idx):
        lr_imgs, hr_imgs = batch
        self.last_imgs = lr_imgs

        # train generator
        if optimizer_idx == 0:
            # generate images
            self.generated_imgs = self(lr_imgs)

            # ground truth result (ie: all fake)
            # put on GPU because we created this tensor inside training_loop
            valid = torch.ones(lr_imgs.size(0), 1)
            if self.on_gpu:
                valid = valid.cuda(lr_imgs.device.index)

            # adversarial loss is binary cross-entropy
            g_loss = self.adversarial_loss(self.discriminator(self.generated_imgs), valid)
            tqdm_dict = {'g_loss': g_loss}
            output = OrderedDict({
                'loss': g_loss,
                'progress_bar': tqdm_dict,
                'log': tqdm_dict
            })
            return output

        # train discriminator
        if optimizer_idx == 1:
            # Measure discriminator's ability to classify real from generated samples

            # how well can it label as real?
            valid = torch.ones(hr_imgs.size(0), 1)
            if self.on_gpu:
                valid = valid.cuda(hr_imgs.device.index)

            real_loss = self.adversarial_loss(self.discriminator(hr_imgs), valid)

            # how well can it label as fake?
            fake = torch.zeros(lr_imgs.size(0), 1)
            if self.on_gpu:
              fake = fake.cuda(lr_imgs.device.index)

            fake_loss = self.adversarial_loss(
                self.discriminator(self.generator(lr_imgs).detach()), fake)

            d_loss = (real_loss + fake_loss) / 2
            tqdm_dict = {'d_loss': d_loss}
            output = OrderedDict({
                'loss': d_loss,
                'progress_bar': tqdm_dict,
                'log': tqdm_dict
            })
            return output

    def configure_optimizers(self):
        opt_g, _ = self.generator.configure_optimizers()
        opt_d = self.discriminator.configure_optimizers()
        return [opt_g[-1], opt_d], []

    def on_epoch_end(self):
        sample_input, _ = next(iter(self.trainer.val_dataloaders[-1]))
        if self.on_gpu:
            sample_input = sample_input.cuda()
        # log sampled images
        sample_imgs = self(sample_input)
        grid = torchvision.utils.make_grid(sample_imgs)
        self.logger.experiment.add_image(f'generated_images', grid, self.current_epoch)
