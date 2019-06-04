import argparse
from src.main_1 import extract_feature

parser = argparse.ArgumentParser()

parser.add_argument("--model_name", type=str,
                    help="""Name of the model.""")

parser.add_argument("--source_path", type=str,
                    help="""Path to source file.""")

parser.add_argument("--target_path", type=str,
                    help="""Path to target file.""")

parser.add_argument("--model_path", type=str,
                    help="""Path to model files.""")

parser.add_argument("--config_path", type=str,
                    help="""Path to config file.""")

parser.add_argument("--batch_size", type=int, default=5,
                    help="""Batch size of beam search.""")

parser.add_argument("--saveto_emb", type=str,
                    help="""Result prefix.""")

parser.add_argument("--saveto_hidden", type=str,
                    help="""Result prefix.""")

parser.add_argument("--saveto_logit", type=str,
                    help="""Result prefix.""")

parser.add_argument("--use_gpu", action="store_true")


def run(**kwargs):
    args = parser.parse_args()

    # Modify some options.
    for k, v in kwargs.items():
        setattr(args, k, v)

    extract_feature(args)


if __name__ == '__main__':
    run()
