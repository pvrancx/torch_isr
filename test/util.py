import argparse
from argparse import Namespace

import torch


def generate_img_batch(n_batch: int, n_channels: int, width: int, height: int):
    tensor = torch.linspace(0., 1., n_batch * width * height * n_channels)
    return tensor.reshape((n_batch, n_channels, height, width))


def generate_hparams(model_type, **kwargs):
    parser = argparse.ArgumentParser()
    parser = model_type.add_model_specific_args(parser)
    hparams = parser.parse_args([])
    dct = vars(hparams)
    dct.update(kwargs)
    return hparams
