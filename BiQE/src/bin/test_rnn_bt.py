import argparse
from src.main_1 import test_rnn_bt
from . import auto_mkdir

parser = argparse.ArgumentParser()

parser.add_argument("--model_name", type=str,
                    help="""Name of the model.""")

parser.add_argument("--source_path", type=str,
                    help="""Path to source file.""")

parser.add_argument("--target_path", type=str,
                    help="""Path to target file.""")

parser.add_argument("--transformer_model_path", type=str,
                    help="""Path to model files.""")

parser.add_argument("--transformer_model_bt_path", type=str,
                    help="""Path to model bt files.""")

parser.add_argument("--qe_model_path", type=str,
                    help="""Path to model files.""")

parser.add_argument("--config_path", type=str,
                    help="""Path to config file.""")

parser.add_argument("--batch_size", type=int, default=5,
                    help="""Batch size of beam search.""")

parser.add_argument("--saveto", type=str,
                    help="""Result prefix.""")

parser.add_argument("--use_gpu", action="store_true")


def run(**kwargs):
    args = parser.parse_args()

    # Modify some options.
    for k, v in kwargs.items():
        setattr(args, k, v)

    auto_mkdir('/'.join(args.saveto.split('/')[0:2]))
    test_rnn_bt(args)


if __name__ == '__main__':
    run()
