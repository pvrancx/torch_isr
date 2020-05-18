import os
from typing import Tuple, Union

from PIL import Image
from torchvision.datasets import VisionDataset
from torchvision.datasets.folder import is_image_file
from torchvision.transforms import RandomCrop, Resize, CenterCrop, transforms


class ImagesFromFolder(VisionDataset):
    """
    Loads images from folder to create dataset. Input and target are equal to loaded image
    (possibly with different transforms applied).
    """
    def __init__(self, root: str, transform=None, target_transform=None):
        super(ImagesFromFolder, self).__init__(root,
                                               transform=transform,
                                               target_transform=target_transform)
        self.images = [os.path.join(root, x) for x in os.listdir(root) if is_image_file(x)]

    def __getitem__(self, index):
        img = Image.open(self.images[index]).convert('RGB')
        target = img.copy()
        if self.transform:
            img = self.transform(img)
        if self.target_transform:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.images)


class IsrDataset(VisionDataset):
    """
    Wraps VisionDataset to create image super resolution dataset. Targets are image crops, inputs
    are targets scaled by given factor.
    """
    def __init__(
            self,
            wrapped_dataset: VisionDataset,
            output_size: Union[int, Tuple[int, int]],
            scale_factor: int = 2,
            deterministic: bool = False,
            base_image_transform=None,
            transform=None,
            target_transform=None
    ):
        super(IsrDataset, self).__init__(wrapped_dataset.root,
                                         transform=transform,
                                         target_transform=target_transform)

        assert scale_factor > 1, 'scale factor must be >=  2'
        if isinstance(output_size, tuple):
            input_size = (output_size[0] // scale_factor, output_size[1] // scale_factor)
        elif isinstance(output_size, int):
            input_size = (output_size // scale_factor, output_size // scale_factor)
        else:
            raise RuntimeError('Invalid output size')

        self._dataset = wrapped_dataset
        self.base_img_transform = base_image_transform
        self._crop = CenterCrop(size=output_size) if deterministic else RandomCrop(size=output_size)
        self._scaler = Resize(size=input_size)

    def __len__(self):
        return len(self._dataset)

    def __getitem__(self, item):
        base_img = self._dataset[item][0]
        if self.base_img_transform is not None:
            base_img = self.base_img_transform(base_img)

        target = self._crop(base_img)
        img = self._scaler(target)

        if self.target_transform is not None:
            target = self.target_transform(target)
        if self.transform is not None:
            img = self.transform(img)
        return img, target

