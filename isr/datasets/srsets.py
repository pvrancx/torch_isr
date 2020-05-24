import os
import zipfile

from torchvision.datasets.utils import download_url

from isr.datasets.isr import ImagesFromFolder


def load_set(
        root: str,
        download: bool = True,
        url=None,
        basename=None,
        image_mode: str = 'RGB',
        transform=None,
        target_transform=None
):

    set_root = os.path.join(root, basename)
    filename = f"{basename}.zip"

    if download:
        assert url is not None, 'URL must be specified for download'
        download_url(url, root, filename=filename, md5=None)
        with zipfile.ZipFile(os.path.join(root, filename), 'r') as zip_ref:
            zip_ref.extractall(root)

    if not os.path.isdir(set_root):
        raise RuntimeError('Dataset not found or corrupted.' +
                           ' You can use download=True to download it')

    return ImagesFromFolder(set_root, transform, target_transform, image_mode=image_mode)


def load_set5(
        root: str,
        download: bool = True,
        image_mode: str = 'RGB',
        transform=None,
        target_transform=None):

    return load_set(
        root=root,
        download=download,
        url='https://drive.google.com/u/0/uc?id=1_JSiLOaNtOmoNSG2_s0bSgL6D4dJyp_8&export=download',
        basename='Set5',
        image_mode=image_mode,
        transform=transform,
        target_transform=target_transform
    )


def load_set14(
        root: str,
        download: bool = True,
        image_mode: str = 'RGB',
        transform=None,
        target_transform=None):
    return load_set(
        root=root,
        download=download,
        url='https://drive.google.com/u/0/uc?id=1nihY72G3ZGVxAIMVVTKOAvFbMYq2GVvg&export=download',
        basename='Set14',
        image_mode=image_mode,
        transform=transform,
        target_transform=target_transform
    )
