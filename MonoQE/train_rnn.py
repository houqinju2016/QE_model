import argparse
import os

from examples.extract_features import train_rnn

parser = argparse.ArgumentParser()
parser.add_argument("--input_file", default=None, type=str, required=True)
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
                        help="The output directory where the model predictions and checkpoints will be written.")

# parser.add_argument("--output_file", default=None, type=str, required=True)
parser.add_argument("--bert_model_src", default=None, type=str, required=True,
                    help="Bert pre-trained model selected in the list: bert-base-uncased, "
                         "bert-large-uncased, bert-base-cased, bert-base-multilingual, bert-base-chinese.")
parser.add_argument("--bert_model_mt", default=None, type=str, required=True,
                    help="Bert pre-trained model selected in the list: bert-base-uncased, "
                         "bert-large-uncased, bert-base-cased, bert-base-multilingual, bert-base-chinese.")
parser.add_argument("--qe_model", default=None, type=str, required=True,
                    help="Bert pre-trained model selected in the list: bert-base-uncased, "
                         "bert-large-uncased, bert-base-cased, bert-base-multilingual, bert-base-chinese.")


parser.add_argument("--task_name",
                        default=None,
                        type=str,
                        required=True,
                        help="The name of the task to train.")
## Other parameters
parser.add_argument("--do_lower_case", default=False, action='store_true',
                    help="Set this flag if you are using an uncased model.")
parser.add_argument("--layers", default="-1,-2,-3,-4", type=str)
parser.add_argument("--max_seq_length", default=128, type=int,
                    help="The maximum total input sequence length after WordPiece tokenization. Sequences longer "
                         "than this will be truncated, and sequences shorter than this will be padded.")
parser.add_argument("--train_batch_size", default=32, type=int, help="Batch size for predictions.")
parser.add_argument("--eval_batch_size", default=32, type=int, help="Batch size for predictions.")
parser.add_argument("--local_rank",
                    type=int,
                    default=-1,
                    help="local_rank for distributed training on gpus")
parser.add_argument("--no_cuda",
                    default=False,
                    action='store_true',
                    help="Whether not to use CUDA when available")

parser.add_argument('--log_path', type=str, default="./log",
                    help="The path for saving tensorboard logs. Default is ./log")
parser.add_argument("--learning_rate",
                        default=5e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
parser.add_argument("--num_train_epochs",
                        default=3.0,
                        type=float,
                        help="Total number of training epochs to perform.")
parser.add_argument("--warmup_proportion",
                        default=0.1,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
parser.add_argument('--fp16',
                        default=False,
                        action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")

def auto_mkdir(path):

    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

def run(**kwargs):
    args = parser.parse_args()

    # Modify some options.
    for k, v in kwargs.items():
        setattr(args, k, v)

    auto_mkdir(args.log_path)
    auto_mkdir(args.output_dir_src)
    auto_mkdir(args.output_dir_mt)

    auto_mkdir(args.output_dir_qe_model)

    train_rnn(args)


if __name__ == '__main__':
    run()
