"""
    Compare two versions of rerankers

    Use them to rank same lists of examples and see how different the 
    results are. 
"""
import argparse
import numpy as np
from metax.reranker_bootstrap.utils import * 
from metax.reranker_bootstrap.gen_better_ds import *
from simpletransformers.classification import ClassificationModel, ClassificationArgs


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
        
def distance_square(ranking1, ranking2):
    pos = {}
    for i in range(len(ranking2)):
        pos[ranking2[i]] = i
    sqr = 0
    for i,e in enumerate(ranking1):
        sqr += (i-pos[e])**2
    return sqr
    

def rank_with_reranker(reranker, tp)
    return reranker_rank(reranker, tp['query_examples'], tp['retrieved_candidates'], just_return_index=True)


def get_parser():
    parser = argparse.ArgumentParser(
        description="compare rerankers")

    parser.add_argument('--checkpoint_A', type=str,
                        help="Reranker checkpoint A")
    parser.add_argument('--checkpoint_B', type=int, default=60,
                        help="Reranker checkpoint B")
    parser.add_argument('--retrieved_data_path', type=str,
                        help="Path to save retrieved and filtered <query, retrieved, tuple_id> tuples to use for comparison")
    parser.add_argument('--reranker_model_type', required=True,
                        type=str, help="Type of reranker model (e.g. bart)")
    parser.add_argument('--reranker_model_name', required=True,
                        type=str, help="Name of reranker model (e.g. facebook/bart-base)")
    return parser



def main():
    args = get_parser().parse_args()
    log_args(args)

    # Load reranker A
    log(f"Loading Reranker A from {args.checkpoint_A}...")
    reranker_A = ClassificationModel(
        args.reranker_model_type, args.checkpoint_A
    )
    log("Reranker A Loaded")

    # Load reranker B
    log(f"Loading Reranker B from {args.checkpoint_B}...")
    reranker_B = ClassificationModel(
        args.reranker_model_type, args.checkpoint_B
    )
    log("Reranker B Loaded")

    data = load_data_from_json(args.retrieve_data_path)
    diffs = []
    for i,tp in enumerate(data):
        log(f"reranking tuple {i+1}/{len(data)}")
        ranking_A = rank_with_reranker(reranker_A, tp)
        ranking_B = rank_with_reranker(reranker_B, tp)
        diff_sqr = distance_square(ranking_A, ranking_B)
        log(f"Squared order difference for tuple {i+1} is: {diff_sqr}")
        diffs.append(diff_sqr)
    avg = np.mean(diffs)
    std = np.std(diffs)
    log(f"Mean: {avg}, Std: {std}")


if __name__ == "__main__":
    main()