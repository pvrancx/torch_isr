# torch_isr
PyTorch models for Image Super Resolution

## Usage
### Training
```
usage: train_isr_model.py [Optional Arguments] {SrResNet,SrCnn,SubPixelSrCnn}

Train Image Super Resolution model

positional arguments:
  {SrResNet,SrCnn,SubPixelSrCnn}
                        type of ISR model to train

optional arguments:
  -h, --help            show help message and exit
  --img_mode {RGB,yCbCr}
                        image mode used by model
  --learning_rate LEARNING_RATE
                        base learning rate
  --lr_epochs LR_EPOCHS
                        epochs over which to decay learning_rate
  --batch_size BATCH_SIZE
                        default train batch size
  --scale_factor SCALE_FACTOR
                        model upscale factor
  --weight_decay WEIGHT_DECAY
                        weight decay penalty (default=0)

 use --help to list other optional arguments

```

Example usage:

```
python train_isr_model.py --gpus 1 --max_epochs 1000 SrCnn
```
### Running
```
usage: super_resolve.py [-h] --model MODEL --checkpoint CHECKPOINT
                        [--output_filename OUTPUT_FILENAME]
                        input_image

Image Super Resolution

positional arguments:
  input_image           input image

required arguments:
  --model MODEL         type of ISR model to use
  --checkpoint CHECKPOINT
                        saved model checkpoint to use

optional arguments:
  -h, --help            show this help message and exit
  --output_filename OUTPUT_FILENAME
                        output image name (default: out.png)

```

Example usage:

```
python super_resolve.py --model SrResNet --checkpoint trained_models/srresnet4x.ckpt input.jpg
```

<p>
  <img src='trained_models/example.png' width='800'/>
</p>
