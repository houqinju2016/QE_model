import argparse
import os

from examples.extract_features import test_rnn

parser = argparse.ArgumentParser()
parser.add_argument("--input_file", default=None, type=str, required=True)
parser.add_argument("--saveto", type=str,
                    help="""Result prefix.""")
parser.add_argument("--bert_model_src", default=None, type=str, required=True,
                    help="Bert pre-trained model selected in the list: bert-base-uncased, "
                         "bert-large-uncased, bert-base-cased, bert-base-multilingual, bert-base-chinese.")
parser.add_argument("--bert_model_mt", default=None, type=str, required=True,
                    help="Bert pre-trained model selected in the list: bert-base-uncased, "
                         "bert-large-uncased, bert-base-cased, bert-base-multilingual, bert-base-chinese.")
parser.add_argument("--task_name",
                        default=None,
                        type=str,
                        required=True,
                        help="The name of the task to train.")
parser.add_argument("--output_dir_src",
                    default=None,
                    type=str,
                    required=True,
                    help="The output directory where the model predictions and checkpoints will be written.")

parser.add_argument("--output_dir_mt",
                    default=None,
                    type=str,
                    required=True,
                    help="The output directory where the model predictions and checkpoints will be written.")
parser.add_argument("--output_dir_qe_model",
                    default=None,
                    type=str,
                    required=True,
                    help="The output directory where the model predictions and checkpoints will be written.")  ## Other parameters

parser.add_argument("--layers", default="-1,-2,-3,-4", type=str)
parser.add_argument("--max_seq_length", default=128, type=int,
                    help="The maximum total input sequence length after WordPiece tokenization. Sequences longer "
                         "than this will be truncated, and sequences shorter than this will be padded.")
parser.add_argument("--test_batch_size", default=32, type=int, help="Batch size for predictions.")

parser.add_argument("--do_lower_case", default=False, action='store_true',
                    help="Set this flag if you are using an uncased model.")

parser.add_argument("--local_rank",
                    type=int,
                    default=-1,
                    help="local_rank for distributed training on gpus")
parser.add_argument("--no_cuda",
                    default=False,
                    action='store_true',
                    help="Whether not to use CUDA when available")


def auto_mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def run(**kwargs):
    args = parser.parse_args()

    # Modify some options.
    for k, v in kwargs.items():
        setattr(args, k, v)

    auto_mkdir('/'.join(args.saveto.split('/')[0:2]))
    test_rnn(args)


if __name__ == '__main__':
    run()
