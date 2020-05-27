from random import random

import torch
from torchvision.datasets import VisionDataset

from isr.datasets.isr import IsrDataset
from isr.models.lightning_model import LightningIsr


class DiscriminatorDataset(VisionDataset):
    def __init__(
            self,
            wrapped_dataset: IsrDataset,
            generator: LightningIsr,
            transform=None,
            target_transform=None
    ):
        super(DiscriminatorDataset, self).__init__(
            wrapped_dataset.root,
            transform=transform,
            target_transform=target_transform
        )
        self._dataset = wrapped_dataset
        self._generator = generator

    def __len__(self):
        return len(self._dataset)

    def __getitem__(self, item):
        lr_img, hr_img = self._dataset[item]

        if random() < 0.5:
            img = self._generator(lr_img.view((1,) + lr_img.shape))[0].detach()
            target = torch.zeros(1, dtype=img.dtype)
        else:
            img = hr_img
            target = torch.ones(1, dtype=img.dtype)

        if self.target_transform is not None:
            target = self.target_transform(target)
        if self.transform is not None:
            img = self.transform(img)
        return img, target
