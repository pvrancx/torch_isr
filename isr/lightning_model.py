from typing import Callable, Any

import torch
from torch.nn import functional as F
from torch import nn
from pytorch_lightning.core.lightning import LightningModule
from torch.optim.lr_scheduler import ReduceLROnPlateau

_default_params = {
    'momentum': 0.9,
    'learning_rate': 1e-3,
    'weight_decay': 1e-6,
    'model_params': {}
}


class LightningIsr(LightningModule):
    def __init__(self, model_factory: Callable[[Any], nn.Module],
                 hparams=None):
        super(LightningIsr, self).__init__()
        params = _default_params.copy()
        params.update(hparams)
        self.hparams = params

        self.model = model_factory(**hparams['model_params'])

    def forward(self, x):
        return self.model(x)

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
        psnr = 10 * torch.log10(1. / mse)
        return {'val_loss': mse, 'val_psnr': psnr}

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
        psnr = 10 * torch.log10(1. / mse)
        return {'test_loss': mse, 'test_psnr': psnr}

    def test_epoch_end(self, outputs):
        avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
        avg_psnr = torch.stack([x['test_psnr'] for x in outputs]).mean()

        tensorboard_logs = {'test_loss': avg_loss, 'test_psnr': avg_psnr}
        return {'test_loss': avg_loss,
                'test_psnr': avg_psnr,
                'log': tensorboard_logs}

    def configure_optimizers(self):
        optim = torch.optim.SGD(
            self.parameters(),
            lr=self.hparams['learning_rate'],
            momentum=self.hparams['momentum'],
            weight_decay=self.hparams['weight_decay']
        )
        scheduler = ReduceLROnPlateau(optim)
        return [optim], [scheduler]
