import os

import pytest
from PIL import Image
import numpy as np

from isr.models import SrResNet, SrCnn, SubPixelSrCnn


@pytest.fixture(params=[1, 3, 32, 64])
def in_channels(request):
    yield request.param


@pytest.fixture(params=[1, 3, 32, 64])
def out_channels(request):
    yield request.param


@pytest.fixture(params=[(9, 4), (32, 32), (15, 64)])
def img_size(request):
    yield request.param


@pytest.fixture(params=[2, 3, 4, 9])
def scale_factor(request):
    yield request.param


@pytest.fixture(params=[SrResNet, SrCnn, SubPixelSrCnn])
def model_type(request):
    yield request.param


@pytest.fixture(scope="session")
def pil_img():
    data = np.arange(3 * 64 * 128).reshape((64, 128, 3))
    return Image.fromarray(data.astype(np.uint8), mode="RGB")


@pytest.fixture(scope="session")
def image_dir(tmpdir_factory, pil_img):
    path = tmpdir_factory.mktemp("data")
    pil_img.save(str(path.join("img1.png")))
    pil_img.save(str(path.join("img2.png")))

    return os.path.abspath(str(path))
