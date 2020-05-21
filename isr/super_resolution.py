import torch
from PIL import Image
from torchvision.transforms import ToTensor, ToPILImage

from isr.models.lightning_model import LightningIsr


def super_resolve_ycbcr(model: LightningIsr, img: Image) -> Image:
    img_ycbcr = img.convert('YCbCr')
    y, cb, cr = img_ycbcr.split()

    to_tensor = ToTensor()
    to_img = ToPILImage(mode='L')

    model_input = to_tensor(y).view(1, -1, y.size[0], y.size[1])
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

    model_input = to_tensor(img_rgb).view(1, -1, img.size[0], img.size[1])
    model_output = model(model_input)[0].cpu().detach() * 255.

    return to_img(model_output.type(torch.uint8))
