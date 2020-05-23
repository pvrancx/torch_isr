from argparse import ArgumentParser

import torch
import torchvision
from pytorch_lightning.core.lightning import LightningModule
from torch.nn import functional as F


def psnr(batch1, batch2):
    reduce_dims = torch.arange(1, batch1.ndim, dtype=torch.int).tolist()
    mse = torch.mean((batch1 - batch2) ** 2, dim=reduce_dims)
    return torch.mean(10 * torch.log10(1. / mse))


class LightningIsr(LightningModule):
    def __init__(self, hparams):
        super(LightningIsr, self).__init__()
        self.hparams = hparams

    @property
    def scale_factor(self) -> int:
        return self.hparams.scale_factor

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--learning_rate', type=float, default=0.002)
        parser.add_argument('--momentum', type=float, default=0.9)
        parser.add_argument('--weight_decay', type=float, default=0.)

        parser.add_argument('--scale_factor', type=int, default=2)
        return parser

    def training_step(self, batch, batch_idx):
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

    def on_epoch_end(self) -> None:
        sample_input = self.trainer.validation_dataloader[-1]
        y_hat = self(sample_input)
        grid = torchvision.utils.make_grid(y_hat)
        self.logger.experiment.add_image(f'generated_images', grid, self.current_epoch)

        current_lr = self.trainer.optimizers[0].param_groups[0]["lr"]
        self.logger.experiment.add_scalar(f'learning_rate', current_lr, self.current_epoch)
