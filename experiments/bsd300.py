from argparse import ArgumentParser

from pytorch_lightning import Trainer
from torch.utils.data import random_split, DataLoader
from torchvision.transforms import transforms

from isr.datasets.bsd import load_bsd300
from isr.datasets.isr import IsrDataset
from isr.models.srcnn import SrCnn, SubPixelSrCnn


def train(args):
    bsd300_train = load_bsd300('../data', split='train')
    bsd300_test = load_bsd300('../data', split='test')

    img_transforms = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
    ])

    train_data = IsrDataset(
        bsd300_train,
        output_size=200,
        scale_factor=args.scale_factor,
        deterministic=False,
        base_image_transform=img_transforms,
        transform=transforms.ToTensor(),
        target_transform=transforms.ToTensor()
    )
    n_train = int(len(train_data) * 0.8)
    split = [n_train, len(train_data) - n_train]
    train_data, val_data = random_split(train_data, split)
    test_data = IsrDataset(
        bsd300_test,
        output_size=200,
        scale_factor=args.scale_factor,
        deterministic=True,
        transform=transforms.ToTensor(),
        target_transform=transforms.ToTensor()
    )

    model = SubPixelSrCnn(hparams=args)
    trainer = Trainer()
    trainer.fit(
        model,
        train_dataloader=DataLoader(test_data, shuffle=True, batch_size=32, num_workers=2),
        val_dataloaders=DataLoader(val_data, shuffle=False, batch_size=32, num_workers=2),
    )
    trainer.test(
        model,
        test_dataloaders=DataLoader(test_data, shuffle=False, batch_size=32, num_workers=2)
    )


if __name__ == '__main__':
    def _main():
        parser = ArgumentParser()
        parser = Trainer.add_argparse_args(parser)

        # figure out which model to use
        parser.add_argument('--model_name', type=str, default='SubPixelSrCnn', help='model name')

        temp_args, _ = parser.parse_known_args()

        # let the model add what it wants
        if temp_args.model_name == 'SrCnn':
            parser = SrCnn.add_model_specific_args(parser)
        elif temp_args.model_name == 'SubPixelSrCnn':
            parser = SubPixelSrCnn.add_model_specific_args(parser)
        else:
            raise RuntimeError('Unknown model')

        args = parser.parse_args()
        train(args)

    _main()




