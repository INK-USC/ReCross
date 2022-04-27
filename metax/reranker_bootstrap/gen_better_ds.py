"""
    Generate one group of better tuple
"""
import argparse
import json
import os
import random
import torch
import logging
import copy
import wandb

from itertools import product
from collections import defaultdict
import numpy as np

from transformers import AutoTokenizer, BartConfig

from metax.models.mybart import MyBart
from metax.task_manager import dataloader
from metax.task_manager.dataloader import GeneralDataset
from metax.distant_supervision.train_reranker import assemble_pairs, create_model_dataframe
from metax.utils import loss_evaluate, model_train, get_optimizer
from metax.models.utils import trim_batch
from metax.reranker_bootstrap.utils import * 

# Hyperparameters
from metax.reranker_bootstrap import config

from simpletransformers.classification import ClassificationModel, ClassificationArgs

# ================ Global Vars ================
# This will be updated by argparse, no need to manually update
MODEL_TYPE = None
main_logger = init_logger()
# =============== Helper Functions =================
def log(msg):
    main_logger.info(msg)


def log_args(args):
    """
        Log initial arguments
    """
    for k, v in vars(args).items():
        main_logger.log(logging.INFO, f"{k:>30} ====> {v}")

def load_reranker(path, model_type):
    """
        Load reranker checkpoint
    """
    log(f"Loading Reranker from {path}...")
    model = ClassificationModel(
        model_type, path
    )
    log("Reranker Loaded")
    return model


def load_model(model_type="yuchenlin/BART0"):
    """
        Load upstream model
    """
    log(f"Loading upstream model from {model_type}...")
    config = BartConfig.from_pretrained(model_type)
    # config.forced_bos_token_id = False
    model = MyBart.from_pretrained(model_type, config=config)
    model.to(torch.device("cuda"))
    log("Upstream model loaded")
    return model

def load_upstream_test_data():
    """
        In funciton adjust_by_eval_loss, we want to use more data than |query| to calculate
        eval loss to make the results more stable. 

        So here load test split of each upstream task

        return:
        -------
        a dict [task] -> [100 instances]
    """
    log("Loading upstream data test splits.")
    upstream_tasks = load_upstream_dataset_list()
    test_data = {}
    for task in upstream_tasks:
        log(f"\nLoading {task} test data")
        # TODO: should we save this sampled 100 instances
        task_data = random.sample(load_data_from_json(f"data/{task}/test-1000.json"),100)
        test_data[task] = task_data
    return test_data


def wrap_dataloader(data):
    """
        Wrap data with a GeneralDataset 
        to feed the data to upstream model

        parameters
        ----------
        data : a list of instances
    """
    pseudo_args = argparse.Namespace(max_input_length=512,
                                     do_lowercase=False,
                                     append_another_bos=True,
                                     max_output_length=512,
                                     train_batch_size=2,
                                     n_gpu=1,
                                     predict_batch_size=2)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_TYPE)
    data_loader = GeneralDataset(
        main_logger,
        pseudo_args,
        None,
        is_training=True,  # Random Sampler
        task_name="Placeholder",
        given_data=data)
    data_loader.load_dataset(tokenizer, skip_cache=True, quiet=True)
    data_loader.load_dataloader()
    return data_loader


# ================ Service Functions =================


def reranker_rank(reranker, queries, sampled_candidates, top_k=0, just_return_index=False):
    """
        Rank sampled_candidates using reranker and return top_k
    """
    assert top_k <= len(
        sampled_candidates), f"top_k value {top_k} exceeds maximum {len(sampled_candidates)}"

    packed_pairs = list(product(queries, sampled_candidates))
    packed_pair_ids = list(
        product(range(len(queries)), range(len(sampled_candidates))))

    packed_pairs = [(q[0], c[0]) for q, c in packed_pairs]
    predictions, logits = reranker.predict(packed_pairs)
    # the [0] is the logit for False, [1] is the logit for True
    scores = [l[1] for l in logits]

    # take the average of the scores
    score_list = defaultdict(list)
    final_score = {}
    for (qid, cid), score in zip(packed_pair_ids, scores):
        score_list[cid].append(score)
    for cid in score_list:
        final_score[cid] = np.mean(score_list[cid])


    if just_return_index:
        reranked_order = [item[1] for item in sorted(zip(sampled_candidates, range(
            len(sampled_candidates))), key=lambda x:final_score[x[1]], reverse=True)]
        if top_k > 0:
            return reranked_order[:top_k]
        return reranked_order

    reranked_retrieved_data = [item[0] for item in sorted(zip(sampled_candidates, range(
        len(sampled_candidates))), key=lambda x:final_score[x[1]], reverse=True)]

    if top_k > 0:
        return reranked_retrieved_data[:top_k]
    return reranked_retrieved_data


def tune_upstream_model(upstream_model, data):
    """
        Tune upstream_model using data. 
    """
    dloader = wrap_dataloader(data)
    total_steps = len(dloader.dataloader)
    optimizer, scheduler = get_optimizer(
        upstream_model, total_steps=total_steps, warmup_steps=3)
    model_train(upstream_model, optimizer, dloader, num_epochs=1, logger=main_logger,
                unfreezed_layers=0, scheduler=scheduler, total_steps=total_steps)


def calculate_loss(upstream_model, data):
    """
        Calculate evaluation loss on data
    """
    log(f"Calculating loss on {len(data)} test examples...")
    dloader = wrap_dataloader(data)
    return loss_evaluate(upstream_model, dloader, gpus=list(range(torch.cuda.device_count())))


def adjust_by_eval_loss(ranked, upstream_model, queries, test_data):
    """
        For each round, we load upstream model
        separate 100 ranked examples into several groups
        fine-tune upstream model on each group and calculate a new loss. 

        Parameters:
        -----------
        ranked : list of top examples from reranked sampled_candidates from reranker (> 1'th iteration)
        upstream_model : BART0 or other upstream models object
        queries : list of query examples 
        test_data : the list of instances for loss calculation (from same task as queries)

        Return: 
        -------
        Reordered list of examples after loss adjustment
    """
    group_sz = 20

    # Calculate a base loss, might need this later
    base_loss = calculate_loss(upstream_model, test_data)
    log(f"Upstream model base loss (without tuning) {base_loss}")

    order = list(range(len(ranked)))  # Used to index ranked
    score_list = defaultdict(list)

    for _ in range(config.ROUNDS):
        log(f"Start round {_}/{config.ROUNDS - 1}")
        # Since ranked are already selected from reranker results
        # The order doesn't matter
        random.shuffle(order)
        groups = np.array_split(order, len(ranked)//group_sz)
        for i, g in enumerate(groups):
            log(f"Start round {_} group {i}")
            group_data = [ranked[i] for i in g]
            model_to_tune = copy.deepcopy(upstream_model)
            tune_upstream_model(model_to_tune, group_data)
            group_loss = calculate_loss(model_to_tune, test_data)
            # Release GPU memory
            del model_to_tune
            log(f"Round {_} group {i} loss after tuning {group_loss}")
            for id in g:
                score_list[id].append(group_loss)

    final_scores = {}
    for id in order:
        final_scores[id] = np.mean(score_list[id])
    # Smaller score means smaller loss
    adjusted_data = [item[0] for item in sorted(zip(ranked, range(
        len(ranked))), key=lambda x:final_scores[x[1]])]

    # TODO: It's possible that all group have loss increases? In that case, maybe we shouldn't have positive examples?

    return adjusted_data


def generate_better_tuple(reranker, upstream_model, queries, sampled_candidates, test_data):
    """
    Generate better distant supervision tuples by reranking and adjust by checking loss effects on upstream model

    Parameters
    ----------
    query : a list of instances. 
    sampled_candidates : a list of 500 instances. 
    """
    # First, rank candidates by current reranker, select top k

    if reranker:
        # TODO: maybe save this order and compare swap distance between iterations to determine convergence.
        log(f"Start reranking {len(sampled_candidates)} sampled_candidates using reranker:")
        ranked = reranker_rank(reranker, queries, sampled_candidates, top_k=config.TOP_K)
    else:
        # If this is first iteration and we don't have reranker yet
        log(f"No reranker initialized, using top {config.TOP_K} examples for loss adjustment. ")
        ranked = sampled_candidates[:config.TOP_K]

    adjusted = adjust_by_eval_loss(ranked, upstream_model, queries, test_data)

    m = len(queries)
    pos = adjusted[:m]
    neg = adjusted[-m:]

    formatted = {'query_examples': queries,
                           'positive_examples': pos, 'negative_examples': neg}
    return formatted


def get_parser():
    parser = argparse.ArgumentParser(
        description="gen_one_group")

    parser.add_argument('--initial_checkpoint_path', type=str,
                        help="Path to initial reranker checkpoint")

    parser.add_argument('--query_and_candidate_data_path', type=str,
                        help="Path to query candidate shard json file generated by bart_retrieve.py")

    parser.add_argument('--data_save_path', type=str, required=True,
                        help="Path to save better tuple generated from loss adjusted ranking")

    parser.add_argument('--upstream_model_type', default="yuchenlin/BART0pp",
                        type=str, help="Upstream model used for bootstrapping instance adjustment")
    parser.add_argument('--reranker_model_type', required=True,
                        type=str, help="Type of reranker model (e.g. bart)")
    parser.add_argument('--reranker_model_name', required=True,
                        type=str, help="Name of reranker model (e.g. facebook/bart-base)")

    return parser


def main():
    args = get_parser().parse_args()
    log_args(args)
    global MODEL_TYPE
    MODEL_TYPE = args.upstream_model_type
    #True if it's first bootstrapping iteration and we don't have reranker yet
    is_first_iteration= True if not args.initial_checkpoint_path else False

    # Create output directories
    make_dir(args.data_save_path)

    wandb.init(project="recross", entity="yuchenlin",
               settings=wandb.Settings(start_method="fork"))

    reranker = load_reranker(args.initial_checkpoint_path, args.reranker_model_type) if not is_first_iteration else None
    upstream_model = load_model(model_type=args.upstream_model_type)
    queries_and_candidates = load_data_from_json(args.query_and_candidate_data_path)
    log(f"Loaded {len(queries_and_candidates)} query/candidates pairs.")

    test_data_dict = load_upstream_test_data()

    better_data = []
    for i,qc_pair in enumerate(queries_and_candidates):
        log(f"Start computing better ds tuple for {i+1}/{len(queries_and_candidates)} query candidate pair. ")
        queries = qc_pair["query_examples"]
        task_name = get_task_from_id(queries[0][2])
        candidates = qc_pair["retrieved_candidates"]
        formatted_tuple = generate_better_tuple(
            reranker, upstream_model, queries, candidates, test_data_dict[task_name])
        better_data.append(formatted_tuple)
    
    save_json(args.data_save_path, "better_tuple.json", better_data)

if __name__ == "__main__":
    main()