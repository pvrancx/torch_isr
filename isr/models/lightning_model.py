import os
from argparse import ArgumentParser

import torch
import torchvision
from pytorch_lightning.core.lightning import LightningModule
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision.transforms import transforms

from isr.datasets.isr import IsrDataset
from isr.datasets.srsets import load_set5, load_set14


def psnr(batch1, batch2):
    reduce_dims = torch.arange(1, batch1.ndim, dtype=torch.int).tolist()
    mse = torch.mean((batch1 - batch2) ** 2, dim=reduce_dims)
    return torch.mean(10 * torch.log10(1. / mse))


class LightningIsr(LightningModule):
    def __init__(self, hparams):
        super(LightningIsr, self).__init__()
        self.hparams = hparams
        self.in_channels = 3 if hparams.img_mode == 'RGB' else 1

    @property
    def scale_factor(self) -> int:
        return self.hparams.scale_factor

    def configure_optimizers(self):
        raise NotImplementedError()

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument(
            '--img_mode',
            type=str, default='RGB', choices=['RGB', 'yCbCr'],
            help='image mode used by model'
        )

        parser.add_argument('--lr_epochs', type=int, default=1000,
                            help='epochs over which to decay learning_rate')
        parser.add_argument('--batch_size', type=int, default=32, help='default train batch size')
        parser.add_argument('--scale_factor', type=int, default=2, help='model upscale factor')
        return parser

    def training_step(self, batch, batch_idx, opt_ind):
        x, y = batch
        y_hat = self(x)
        loss = F.mse_loss(y_hat, y)

        tensorboard_logs = {'train_loss': loss}

        return {'loss': loss, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        mse = F.mse_loss(y_hat, y)
        return {'val_loss': mse, 'val_psnr': psnr(y_hat, y)}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        avg_psnr = torch.stack([x['val_psnr'] for x in outputs]).mean()
        tensorboard_logs = {'val_loss': avg_loss, 'val_psnr': avg_psnr}
        return {'val_loss': avg_loss,
                'val_psnr': avg_psnr,
                'log': tensorboard_logs}

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        mse = F.mse_loss(y_hat, y)
        return {'test_loss': mse, 'test_psnr': psnr(y_hat, y)}

    def test_epoch_end(self, outputs):
        avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
        avg_psnr = torch.stack([x['test_psnr'] for x in outputs]).mean()

        tensorboard_logs = {'test_loss': avg_loss, 'test_psnr': avg_psnr}
        return {'test_loss': avg_loss,
                'test_psnr': avg_psnr,
                'log': tensorboard_logs}

    def train_dataloader(self):
        set5 = IsrDataset(
            load_set5(os.getcwd(), download=True, image_mode=self.hparams.img_mode),
            deterministic=False,
            output_size=8 * self.hparams.scale_factor,
            scale_factor=self.hparams.scale_factor,
            transform=transforms.ToTensor(),
            target_transform=transforms.ToTensor()
        )
        return DataLoader(set5, batch_size=self.hparams.batch_size)

    def val_dataloader(self):
        set5 = IsrDataset(
            load_set5(os.getcwd(), download=True, image_mode=self.hparams.img_mode),
            deterministic=True,
            output_size=228,
            scale_factor=self.hparams.scale_factor,
            transform=transforms.ToTensor(),
            target_transform=transforms.ToTensor()
        )
        return DataLoader(set5, batch_size=self.hparams.batch_size)

    def test_dataloader(self):
        set14 = IsrDataset(
            load_set14(os.getcwd(), download=True, image_mode=self.hparams.img_mode),
            deterministic=True,
            output_size=250,
            scale_factor=self.hparams.scale_factor,
            transform=transforms.ToTensor(),
            target_transform=transforms.ToTensor()
        )
        return DataLoader(set14, batch_size=1)

    def on_epoch_end(self) -> None:
        sample_input, _ = next(iter(self.trainer.val_dataloaders[-1]))
        if self.on_gpu:
            sample_input = sample_input.cuda()
        y_hat = self(sample_input)
        idx = min(4, y_hat.size(0))
        grid = torchvision.utils.make_grid([y_hat[i] for i in range(idx)])
        self.logger.experiment.add_image(f'generated_images', grid, self.current_epoch)

        current_lr = self.trainer.optimizers[0].param_groups[0]["lr"]
        self.logger.experiment.add_scalar(f'learning_rate', current_lr, self.current_epoch)
