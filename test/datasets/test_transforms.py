import pytest

from isr.datasets.transforms import ChannelSelect


@pytest.mark.parametrize("channel_idx", [0, 1, 2])
def test_channel_select(pil_img, channel_idx: int):
    transform = ChannelSelect(channel_idx)
    channels = pil_img.split()

    output = transform(pil_img)

    assert output.mode == "L"
    assert channels[channel_idx] == output
