import os

from torchvision.datasets.voc import download_extract

from isr.datasets.isr import ImagesFromFolder


def load_bsd300(
        root: str,
        download: bool = True,
        split: str = 'train',
        image_mode: str = 'RGB',
        transform=None,
        target_transform=None
):
    assert split.lower() in ['train', 'test'], 'unknown dataset split'

    bsd_root = os.path.join(root, "BSDS300")
    image_dir = os.path.join(bsd_root, 'images')

    if download:
        url = 'http://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/BSDS300-images.tgz'
        download_extract(url, root, os.path.basename(url), None)

    if not os.path.isdir(bsd_root):
        raise RuntimeError('Dataset not found or corrupted.' +
                           ' You can use download=True to download it')

    split_dir = os.path.join(image_dir, split)

    return ImagesFromFolder(split_dir, transform, target_transform, image_mode=image_mode)
