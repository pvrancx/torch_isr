import pytest
import torch

from isr.models.srgan import Discriminator, SrGan
from test.util import generate_hparams, generate_img_batch


def test_discriminator_train_step():
    """Check we can perform train step on model"""
    img_size = (96, 96)
    hparams = generate_hparams(Discriminator, img_shape=img_size)
    model = Discriminator(hparams)
    inputs = generate_img_batch(1, 3, img_size[1], img_size[0])
    targets = torch.ones(inputs.size(0)).view(-1, 1)

    output = model.training_step((inputs, targets), 0)

    assert 'loss' in output


def test_discriminator_val_step():
    """Check we can perform validation step on model"""
    img_size = (96, 96)
    hparams = generate_hparams(Discriminator, img_shape=img_size)
    model = Discriminator(hparams)
    inputs = generate_img_batch(1, 3, img_size[1], img_size[0])
    targets = torch.ones(inputs.size(0)).view(-1, 1)

    output = model.validation_step((inputs, targets), 0)

    assert 'val_loss' in output


@pytest.mark.parametrize("opt_index", [0, 1])
def test_srgan_train_step(scale_factor: int, opt_index: int):
    """Check we can perform train step on gan model"""
    img_size = (48, 48)
    output_size = (img_size[0] * scale_factor, img_size[1] * scale_factor)

    hparams = generate_hparams(SrGan, img_shape=output_size, scale_factor=scale_factor)
    model = SrGan(hparams)

    inputs = generate_img_batch(1, hparams.in_channels, img_size[1], img_size[0])
    targets = generate_img_batch(1, hparams.in_channels, img_size[1] * scale_factor,
                                 img_size[0] * scale_factor)

    output = model.training_step((inputs, targets), 0, opt_index)

    assert 'loss' in output
