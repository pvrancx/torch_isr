from argparse import ArgumentParser

from pytorch_lightning import Trainer


def main():
    parser = ArgumentParser(description='Train Image Super Resolution model')
    parser.add_argument(
        'model_type', type=str,
        choices=['SrResNet', 'SrCnn', 'SubPixelSrCnn'],
        help='type of ISR model to train'
    )
    parser = Trainer.add_argparse_args(parser)

    temp_args, _ = parser.parse_known_args()
    module = __import__('isr.models', fromlist=[temp_args.model_type])
    model_class = getattr(module, temp_args.model_type)

    parser = model_class.add_model_specific_args(parser)
    args = parser.parse_args()

    trainer = Trainer.from_argparse_args(args)

    model = model_class(args)
    trainer.fit(model)


if __name__ == '__main__':
    main()
