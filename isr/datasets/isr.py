import os
from typing import Tuple, Union, Optional

from PIL import Image
from torchvision.datasets import VisionDataset
from torchvision.datasets.folder import is_image_file
from torchvision.transforms import RandomCrop, CenterCrop
from torchvision.transforms.functional import resize

_PIL_IMAGE_MODES_ = ('L', 'F', 'I', 'HSV', 'LAB', 'RGB', 'YCbCr', 'CMYK', 'RGBA', '1')


class ImagesFromFolder(VisionDataset):
    """
    Loads images from folder to create dataset. Input and target are equal to loaded image
    (possibly with different transforms applied).
    """
    def __init__(self, root: str, transform=None, target_transform=None, image_mode: str = 'RGB'):
        super(ImagesFromFolder, self).__init__(root,
                                               transform=transform,
                                               target_transform=target_transform)

        assert image_mode in _PIL_IMAGE_MODES_, 'Unknown PIL image mode.'
        assert os.path.isdir(root), 'Image folder not found.'

        self.images = [os.path.join(root, x) for x in os.listdir(root) if is_image_file(x)]
        self.mode = image_mode

    def __getitem__(self, index):
        img = Image.open(self.images[index]).convert(self.mode)
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
            scale_factor: int = 2,
            output_size: Optional[Union[int, Tuple[int, int]]] = None,
            deterministic: bool = False,
            scale_mode: int = Image.BICUBIC,
            base_image_transform=None,
            transform=None,
            target_transform=None
    ):
        super(IsrDataset, self).__init__(wrapped_dataset.root,
                                         transform=transform,
                                         target_transform=target_transform)

        assert scale_factor > 1, 'scale factor must be >=  2'
        self.scale_factor = scale_factor
        self.scale_mode = scale_mode
        self._dataset = wrapped_dataset
        self.base_img_transform = base_image_transform

        if output_size is None:
            self._crop = None
        elif deterministic:
            self._crop = CenterCrop(size=output_size)
        else:
            self._crop = RandomCrop(size=output_size)

    def _scale(self, img: Image) -> Image:
        width, height = img.size[0] // self.scale_factor, img.size[1] // self.scale_factor
        return resize(img, (height, width), self.scale_mode)

    def __len__(self):
        return len(self._dataset)

    def __getitem__(self, item):
        base_img = self._dataset[item][0]
        if self.base_img_transform is not None:
            base_img = self.base_img_transform(base_img)

        target = self._crop(base_img) if self._crop else base_img
        img = self._scale(target)

        if self.target_transform is not None:
            target = self.target_transform(target)
        if self.transform is not None:
            img = self.transform(img)
        return img, target

