from argparse import ArgumentParser

import torch
from PIL import Image
from torchvision.transforms import ToTensor, ToPILImage

from isr.models.lightning_model import LightningIsr


def super_resolve_ycbcr(model: LightningIsr, img: Image) -> Image:
    img_ycbcr = img.convert('YCbCr')
    y, cb, cr = img_ycbcr.split()

    to_tensor = ToTensor()
    to_img = ToPILImage(mode='L')

    model_input = to_tensor(y).view(1, -1, y.size[1], y.size[0])  # PIL size is (width, height)
    model_output = model(model_input)[0].cpu().detach() * 255.

    y_scaled = to_img(model_output.type(torch.uint8))
    cb_scaled = cb.resize(y_scaled.size, Image.BICUBIC)
    cr_scaled = cr.resize(y_scaled.size, Image.BICUBIC)

    scaled_img = Image.merge('YCbCr', [y_scaled, cb_scaled, cr_scaled]).convert('RGB')
    return scaled_img


def super_resolve_rgb(model: LightningIsr, img: Image) -> Image:
    img_rgb = img.convert('RGB')

    to_tensor = ToTensor()
    to_img = ToPILImage(mode='RGB')

    model_input = to_tensor(img_rgb).view(1, -1, img.size[1], img.size[0])
    model_output = model(model_input)[0].cpu().detach() * 255.

    return to_img(model_output.type(torch.uint8))


def _main():
    parser = ArgumentParser(description='Image Super Resolution')

    parser.add_argument('input_image', type=str, help='input image')
    parser.add_argument('--model', type=str,
                        choices=['SrResNet', 'SrCnn', 'SubPixelSrCnn'],
                        required=True,
                        help='type of ISR model to use')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='saved model checkpoint to use')
    parser.add_argument('--output_filename', type=str, default='out.png',
                        help='output image name (default: out.png)')

    args = parser.parse_args()

    img = Image.open(args.input_image)
    module = __import__('isr.models', fromlist=[args.model])
    model_class = getattr(module, args.model)
    model = model_class.load_from_checkpoint(args.checkpoint)

    if model.hparams.in_channels == 3:
        out = super_resolve_rgb(model, img)
    elif model.hparams.in_channels == 1:
        out = super_resolve_ycbcr(model, img)
    else:
        raise RuntimeError('Unknown image mode: '+args.mode)

    out.save(args.output_filename)


if __name__ == '__main__':
    _main()
