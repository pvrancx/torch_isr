class ChannelSelect:
    """Returns single channel from given multi-channel PIL Image.

    Args:
        channel_idx (int): Desired output channel e.g. 0,1,2 for 3 channel RGB image.
    """

    def __init__(self, channel_idx: int = 0):
        assert 0 < channel_idx < 5, 'Channel index must be int in [0, 1, 2, 3]'
        self.channel_idx = channel_idx

    def __call__(self, img):
        """
        Args:
            img (PIL Image)

        Returns:
            PIL Image: Single channel PIL image containing only selected channel
        """
        channels = img.split()
        return channels[self.channel_idx]

    def __repr__(self):
        return self.__class__.__name__ + '(channel_idx={0})'.format(self.channel_idx)
