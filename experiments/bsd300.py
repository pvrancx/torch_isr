from pytorch_lightning import Trainer
from torch.utils.data import random_split, DataLoader
from torchvision.transforms import transforms

from isr.datasets.bsd import load_bsd300
from isr.datasets.isr import IsrDataset
from isr.lightning_model import LightningIsr
from isr.models.srcnn import SrCnn


def main():
    bsd300_train = load_bsd300('../data', split='train')
    bsd300_test = load_bsd300('../data', split='test')

    train_transforms = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor()
    ])

    test_transforms = transforms.ToTensor()
    train_data = IsrDataset(
        bsd300_train,
        output_size=200,
        scale_factor=2,
        deterministic=False,
        transform=train_transforms,
        target_transform=transforms.ToTensor()
    )
    n_train = int(len(train_data) * 0.8)
    split = [n_train, len(train_data) - n_train]
    train_data, val_data = random_split(train_data, split)
    test_data = IsrDataset(
        bsd300_test,
        output_size=200,
        scale_factor=2,
        deterministic=True,
        transform=test_transforms,
        target_transform=transforms.ToTensor()
    )

    model = LightningIsr(SrCnn, {})
    trainer = Trainer()
    trainer.fit(
        model,
        train_dataloader=DataLoader(train_data, shuffle=True, batch_size=32, num_workers=2),
        val_dataloaders=DataLoader(val_data, shuffle=False, batch_size=32, num_workers=2),
    )
    trainer.test(
        model,
        test_dataloaders=DataLoader(test_data, shuffle=False, batch_size=32, num_workers=2)
    )


if __name__ == '__main__':
    main()




