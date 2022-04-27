from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from metax.models.run_bart import run
import torch
import numpy as np
import random
import logging
import argparse
from metax.models.utils import set_seeds, pickle_load


# Enable import metax
import sys
import os
sys.path.insert(0, ".")


class BaseParser(argparse.ArgumentParser):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # Basic parameters
        def str2bool(a: str):
            return a.lower() == 'true' or a.lower() == '1'

        self.add_argument("--run_name", default="debug", type=str)

        self.add_argument("--train_file", default="data", required=False)
        self.add_argument("--override_train_file",
                          default=None, required=False)
        self.add_argument("--dev_file", default="data", required=False)
        self.add_argument("--test_file", default="data", required=False)
        self.add_argument("--dataset", default="None", required=False)
        self.add_argument("--log_dir", default=None, required=False)
        self.add_argument("--upstream_train_file", default="data/bart0_upstream_train_lines.json", required=False)
        self.add_argument("--memory_cache_path", default="memory_cache/random_memory.pkl", required=False)


        self.add_argument("--query_aggregation_mode", default="aggregate_choices", required=False,
                            help="How to aggregate query examples for querying retrieved data. [aggregate_scores, aggregate_choices].")
        self.add_argument("--model",
                          default="facebook/bart-base",
                          required=False)

        self.add_argument("--output_dir", default="outputs",
                          type=str, required=False)
        self.add_argument("--do_train", action='store_true')
        self.add_argument("--do_predict", action='store_true')
        self.add_argument("--predict_checkpoint",
                          type=str,
                          default="best-model.pt")

        self.add_argument("--n_gpu",
                          type=int,
                          default=torch.cuda.device_count(),
                          required=False)

        # Indexing-related
        self.add_argument("--num_shards_for_indexing",
                          type=int,
                          default=1,
                          required=False)
        self.add_argument("--shard_id_for_indexing",
                          type=int,
                          default=0,
                          required=False)

        # Model parameters
        self.add_argument("--model_type", type=str,
                          default='facebook/bart-base')
        self.add_argument("--model_path", type=str, default=None)
        self.add_argument("--checkpoint", type=str, default=None)
        self.add_argument("--checkpoint_prefix", type=str, default="unnamed_checkpoint-")
        self.add_argument("--do_lowercase", action='store_true', default=False)
        self.add_argument("--freeze_embeds",
                          action='store_true', default=False)

        # Preprocessing/decoding-related parameters
        self.add_argument('--max_input_length', type=int, default=512)
        self.add_argument('--max_input_length_for_eval', type=int, default=512)
        self.add_argument('--max_output_length', type=int, default=64)
        self.add_argument('--num_beams', type=int, default=4)
        self.add_argument("--append_another_bos", type=str2bool, default="True")    # This is very important

        # Training-related parameters
        self.add_argument("--train_batch_size",
                          default=64,
                          type=int,
                          help="Batch size per GPU/CPU for training.")
        self.add_argument("--predict_batch_size",
                          default=32,
                          type=int,
                          help="Overall predict batch size for evaluation.")
        self.add_argument("--learning_rate",
                          default=3e-5,
                          type=float,
                          help="The initial learning rate for Adam.")
        self.add_argument("--ret_learning_rate",
                          default=3e-6,
                          type=float,
                          help="The initial learning rate for Adam.")

        self.add_argument("--train_seed",
                          default=3331,
                          type=int,
                          help="Random seed for training.")
        # self.add_argument("--warmup_proportion", default=0.01, type=float,
        #                     help="Weight decay if we apply some.")    # Not used
        self.add_argument("--weight_decay",
                          default=0.01,
                          type=float,
                          help="Weight deay if we apply some.")
        self.add_argument("--adam_epsilon",
                          default=1e-8,
                          type=float,
                          help="Epsilon for Adam optimizer.")
        self.add_argument("--max_grad_norm",
                          default=0.1,
                          type=float,
                          help="Max gradient norm.")
        self.add_argument("--gradient_accumulation_steps",
                          default=1,
                          type=int,
                          help="Max gradient norm.")
        self.add_argument("--num_train_epochs",
                          default=1.0,
                          type=float,
                          help="Total number of training epochs to perform.")
        self.add_argument("--num_train_epochs_fs",
                          default=1.0,
                          type=float,
                          help="Total number of training epochs to perform.")
        self.add_argument("--warmup_steps",
                          default=-1,
                          type=int,
                          help="Linear warmup over warmup_steps.")
        self.add_argument("--total_steps",
                          default=-1,
                          type=int,
                          help="Linear warmup over warmup_steps.")
        self.add_argument('--saving_steps', type=int, default=100)
        self.add_argument('--wait_step', type=int, default=10)

        # Other parameters
        self.add_argument("--quiet",
                          action='store_true',
                          help="If true, tqdm will not show progress bar")
        self.add_argument('--eval_period',
                          type=int,
                          default=100,
                          help="Evaluate & save model")
        self.add_argument('--prefix',
                          type=str,
                          default='',
                          help="Prefix for saving predictions")
        self.add_argument('--debug',
                          action='store_true',
                          help="Use a subset of data for debugging")
        self.add_argument('--seed',
                          type=int,
                          default=42,
                          help="random seed for initialization")
        self.add_argument('--save_upstream',
                          action='store_true',
                          help="whether or not save upstream model")
        self.add_argument('--use_retriever',
                          action='store_true',
                          help="whether or not to use the trained retriever")
        self.add_argument('--retriever_mode',
                          default='Random',
                          help="The retriever mode for additional example. "
                          + "One of ['Random', 'Trained', 'SentenceTransformer']")
        self.add_argument('--ret_merge_mode',
                          default='mix',
                          help="The retriever mode for additional example. "
                          + "One of ['mix', 'two-stage'")
        self.add_argument('--retrieve_seed',
                          type=int,
                          default=3331,
                          help="Random seed for retrieving upstream data.")
        self.add_argument("--memory_encoder_path", type=str, required=False)
        self.add_argument("--query_encoder_path", type=str, required=False)

        self.add_argument('--target_training_path',
                          type=str,
                          default='',
                          help="path storing samples used for target few-shot training")

        self.add_argument(
            "--num_upstream_samples",
            type=int,
            default=1000,
            help="number of samples taken in each upstream dataset in upstream learning")
        self.add_argument(
            "--num_shots",
            type=int,
            default=32,
            help="number of samples taken in each downstream dataset in downstream fine-tuning",
        )
        self.add_argument(
            "--upstream_num_shots",
            type=int,
            default=None,
            help="number of samples taken in each downstream dataset in downstream fine-tuning",
        )
        self.add_argument(
            "--num_rounds",
            type=int,
            default=10,
            help="run downstream fine-tuning and evaluating for num_round times",
        )
        self.add_argument(
            "--retrieval_sample_percent",
            type=int,
            default=100,
            help="Percentage of test set to use as query examples",
        )
        self.add_argument(
            "--use_random_retrieval_queries",
            type=bool,
            default=False,
            help="Whether or not to select query examples from the test set randomly",
        )
        self.add_argument(
            "--action",
            type=str,
            default='upstream_training',
            help="action of the pipeline",
        )
        self.add_argument(
            "--target_task",
            type=str,
            default='',
            help="select single task to finetune",
        )
        self.add_argument(
            "--data_dir",
            type=str,
            default='/home/beiwen/MetaCross/data',
            help="the datasets base directory",
        )
        self.add_argument(
            "--retrieved_data_dir",
            type=str,
            default="retrieved_data",
            help="directory to save data retrieved by retriever",
        )
        self.add_argument(
            "--finetune_round",
            type=int,
            default=3,
            help="number of round to do finetune",
        )
        self.add_argument(
            "--finetune_layers",
            type=int,
            default=0,
            help="number of T5Block to fine-tune. If passed N=0, all layers are fine-tuned, otherwise only the last N T5Block of encoder and decoder is finetuned",
        )


        # reranking related 

        self.add_argument(
            "--reranker_model_path",
            type=str,
            default="",
            help="directory to save data retrieved by retriever",
        )

        self.add_argument(
            "--reranker_oversample_rate",
            type=float,
            default=1,
            help="upstream_num_shots *= this",
        )

        self.add_argument(
            "--reranker_query_size",
            type=int,
            default=-1,
            help="upstream_num_shots * this = the infer size",
        )


        # abalation study remove one group of tasks from retrieved examples
        self.add_argument("--remove_group", type=str, default="none")
        self.add_argument("--grouping_config_file", type=str, default="metax/distant_supervision/task_groupings.json")


        self.add_argument("--early_stopping", type=str2bool)
        self.add_argument("--save_checkpoint", type=str2bool)
        self.add_argument("--test_on_the_fly", type=str2bool, default="True")

def get_parser():
    return BaseParser()


def main():
    args = get_parser().parse_args()
    # if os.path.exists(args.output_dir) and os.listdir(args.output_dir):
    #     print("Output directory () already exists and is not empty.")
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)

    # Start writing logs

    log_filename = "{}log.txt".format("train_" if args.do_train else "eval_")

    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
        datefmt='%m/%d/%Y %H:%M:%S',
        level=logging.INFO,
        handlers=[
            logging.FileHandler(os.path.join(args.output_dir, log_filename)),
            logging.StreamHandler()
        ])
    logger = logging.getLogger(__name__)
    logger.info(args)
    logger.info(args.output_dir)

    set_seeds(args.seed)
    args.n_gpu = torch.cuda.device_count()

    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    if not args.do_train and not args.do_predict:
        raise ValueError(
            "At least one of `do_train` or `do_predict` must be True.")

    if args.do_train:
        if not args.train_file:
            raise ValueError(
                "If `do_train` is True, then `train_dir` must be specified.")
        if not args.dev_file:
            raise ValueError(
                "If `do_train` is True, then `predict_dir` must be specified.")

    if args.do_predict:
        if not args.test_file:
            raise ValueError(
                "If `do_predict` is True, then `predict_dir` must be specified."
            )

    logger.info("Using {} gpus".format(args.n_gpu))
    given_data = None
    if args.override_train_file is not None:
        logger.info(f"Using override train file {args.override_train_file}")
        given_data = pickle_load(args.override_train_file)
    run(args, logger, given_data)


if __name__ == '__main__':
    main()
