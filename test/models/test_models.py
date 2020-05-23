from typing import Tuple

from test.util import generate_hparams, generate_img_batch


def test_output_scale(model_type, scale_factor: int, img_size: Tuple[int, int]):
    """Check model output has corrext dimensions"""

    hparams = generate_hparams(model_type, scale_factor=scale_factor)
    model = model_type(hparams)
    batch = generate_img_batch(1, 3, img_size[1], img_size[0])
    output = model(batch)

    assert output.size(2) == img_size[0] * scale_factor
    assert output.size(3) == img_size[1] * scale_factor


def test_train_step(model_type, scale_factor: int):
    """Check we can perform train step on model"""
    img_size = (16, 16)
    hparams = generate_hparams(model_type, scale_factor=scale_factor)
    model = model_type(hparams)
    inputs = generate_img_batch(1, 3, img_size[1], img_size[0])
    targets = generate_img_batch(1, 3, img_size[1] * scale_factor, img_size[0] * scale_factor)

    output = model.training_step((inputs, targets), 0)

    assert 'loss' in output


def test_val_step(model_type, scale_factor: int):
    """Check we can perform validation step on model"""
    img_size = (16, 16)
    hparams = generate_hparams(model_type, scale_factor=scale_factor)
    model = model_type(hparams)
    inputs = generate_img_batch(1, 3, img_size[1], img_size[0])
    targets = generate_img_batch(1, 3, img_size[1] * scale_factor, img_size[0] * scale_factor)

    output = model.validation_step((inputs, targets), 0)

    assert 'val_loss' in output
