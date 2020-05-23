import pytest

from isr.models.srresnet import _SubPixelBlock, _ResNetModule, _ScaleModule
from test.util import generate_img_batch


@pytest.fixture(params=[_SubPixelBlock, _ResNetModule, _ScaleModule])
def module_type(request):
    yield request.param


def test_output_has_correct_num_channels(module_type, in_channels, out_channels):
    module = module_type(in_channels=in_channels, out_channels=out_channels)
    batch = generate_img_batch(1, in_channels, 16, 16)
    output = module(batch)

    assert output.size(1) == out_channels

