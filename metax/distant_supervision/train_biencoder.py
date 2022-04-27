import sys
sys.path.insert(0, "../..")
from metax.meta_algs.biencoder_index import TrainableIndexManager
import json
import logging
import random
import argparse
import os
import torch

os.environ['TOKENIZERS_PARALLELISM'] = 'False'

DEFAULT_RUNNAME = random.randrange(1,100)

# Paths
DS_TRAIN_DATA_PATH = "data/ds_from_bart0_upstream_train.json"
# n.b. -- test data is group holdout, while dev file is just train task data holdout
DS_EVAL_DATA_PATH = "data/ds_from_bart0_upstream_test.json"
BIENCODER_MODEL_PATH = "checkpoints/biencoder_bart0/"
PARTITIONED_DATA_PATH= "data/biencoder_copy/"


main_logger = logging.getLogger("data_preprocess")
main_logger.setLevel(logging.INFO)

def str2bool(a: str):
    return a.lower() == 'true' or a.lower() == '1'

def make_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def save_copy_of_data(train_data, eval_data, args):
    train_data_path = args.partitioned_data_path + "train.json"
    eval_data_path = args.partitioned_data_path + "eval.json"
    json.dump(train_data, open(train_data_path, 'w'), indent=4)
    json.dump(eval_data, open(eval_data_path, 'w'), indent=4)

def prep_data(args):
    """
        Load raw ds data and prepare dataset for training
    """
    train_rdata = json.load(open(args.ds_train_data_path))
    eval_rdata = json.load(open(args.ds_eval_data_path))

    train_data = [(item["query_examples"], item["positive_examples"], item["negative_examples"]) for item in train_rdata]
    eval_data = [(item["query_examples"], item["positive_examples"], item["negative_examples"]) for item in eval_rdata]

    random.seed(42)
    random.shuffle(train_data)
    random.shuffle(eval_data)

    save_copy_of_data(train_data, eval_data, args)

    main_logger.info(f"Training Dataset Size {len(train_data)} | Evalutation Dataset Size {len(eval_data)}")

    return train_data, eval_data


def get_parser():
    parser = argparse.ArgumentParser(description="Traine Reranker")
    parser.add_argument("--runname", type=str, default=f"Run_{DEFAULT_RUNNAME}", help="Name of run")
    parser.add_argument('--biencoder_model_path', type=str, default=BIENCODER_MODEL_PATH, help="Path prefix to store the checkpoints under.")

    # Parameters for training
    parser.add_argument('--ds_train_data_path', type=str, default=DS_TRAIN_DATA_PATH, help="Path to distant supervision data (Training).")
    parser.add_argument('--ds_eval_data_path', type=str, default=DS_EVAL_DATA_PATH, help="Path to distant supervision data (Evaluation).")
    parser.add_argument('--partitioned_data_path', type=str, default=PARTITIONED_DATA_PATH, help="Where to save a copy of the partitioned train/eval data.")
    parser.add_argument("--do_lowercase", action='store_true', default=False,help="train all as lowercase")
    parser.add_argument("--use_cuda", action='store_true', default=True,help="train on GPU")

    # Preprocessing/decoding-related parameters
    # TODO -- update all these
    parser.add_argument('--max_input_length', type=int, default=512)
    parser.add_argument('--max_output_length', type=int, default=64)
    parser.add_argument('--num_beams', type=int, default=4)
    parser.add_argument('--learning_rate', type=float, default=4e-3)
    parser.add_argument("--append_another_bos",
                        type=str2bool,
                        default=True)

    # Training-related parameters
    parser.add_argument("--train_batch_size", default=8, type=int, help="Batch size per GPU/CPU for training.")
    parser.add_argument("--predict_batch_size", default=8, type=int, help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument("--n_epochs", default=16, type=int, help="Number of epochs to train for.")
    parser.add_argument("--n_gpu", default=1, type=int, help="GPU count.")
    parser.add_argument("--model_type", default="yuchenlin/BART0", type=str, help="Type of the model.")
    parser.add_argument("--ret_merge_mode", default="unsupervised", type=str, help="Type of the merge mode of retrieved examples.")

    return parser

def main():
    args = get_parser().parse_args()
    make_dir(args.biencoder_model_path)
    make_dir(PARTITIONED_DATA_PATH)

    logging.basicConfig(level=logging.INFO)
    transformers_logger = logging.getLogger("transformers")
    transformers_logger.setLevel(logging.WARNING)
    main_logger.info(f"--------------------------{args.runname}--------------------------")
    train_data, eval_data = prep_data(args)

    # Optional model configuration
    model = TrainableIndexManager(main_logger, args)
    model.train_biencoder(train_data, eval_data)


if __name__ == "__main__":
    main()
