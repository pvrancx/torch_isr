from typing import Tuple

from isr.datasets.isr import ImagesFromFolder, IsrDataset


def test_images_from_folder(image_dir):
    """Check we can load images from folder as dataset"""

    dataset = ImagesFromFolder(image_dir)
    assert len(dataset) == 2


def test_images_input_target(image_dir):
    """Check image and target are identical"""

    dataset = ImagesFromFolder(image_dir)
    img1, img2 = dataset[-1]
    assert img1 == img2


def test_isr_dataset(image_dir):
    dataset = ImagesFromFolder(image_dir)
    isr_dataset = IsrDataset(dataset)
    assert len(isr_dataset) == 2


def test_isr_dataset_scale(image_dir: str, scale_factor: int):
    """ Check lr input / hr target have correct scale factor"""

    dataset = ImagesFromFolder(image_dir)
    isr_dataset = IsrDataset(dataset, scale_factor=scale_factor)
    img1, img2 = isr_dataset[-1]
    assert img1.size[0] == img2.size[0] // scale_factor
    assert img1.size[1] == img2.size[1] // scale_factor


def test_isr_dataset_output_size(image_dir: str, img_size: Tuple[int, int]):
    """ Check target has correct size """

    dataset = ImagesFromFolder(image_dir)
    isr_dataset = IsrDataset(dataset, output_size=img_size)
    _, img = isr_dataset[-1]
    assert img.size[0] == img_size[1]  # PIL image uses (width, height)
    assert img.size[1] == img_size[0]


def test_isr_dataset_deterministic(image_dir: str, img_size: Tuple[int, int]):
    """ Check deterministic returns the same target crop  """

    dataset = ImagesFromFolder(image_dir)
    isr_dataset = IsrDataset(dataset, output_size=img_size, deterministic=True)
    _, img1 = isr_dataset[-1]
    _, img2 = isr_dataset[-1]

    assert img1 == img2
