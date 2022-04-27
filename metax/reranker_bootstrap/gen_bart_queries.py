"""
    Generate queries used by bart retriever

    Randomly select a group then select query
    All groups have equal chances of appearing in query (independent of group size). 
"""
import argparse

import logging
import random
from typing import List
from metax.reranker_bootstrap.utils import *
from collections import defaultdict

# Task grouping meta-data
from metax.distant_supervision.ds_gen import GROUPING_PATH, load_grouping

main_logger = init_logger()

DS_DATA_PATH = "data/ds_from_bart0_upstream_train.json"

def log_args(args):
    """
        Log initial arguments
    """
    for k, v in vars(args).items():
        main_logger.log(logging.INFO, f"{k:>30} ====> {v}")
def log(msg):
    main_logger.info(msg)


def get_parser():
    parser = argparse.ArgumentParser(
        description="retrieve with bart")

    parser.add_argument('--N', type=int, default=500,
                        help="How many query files to generate.")
    parser.add_argument('--output_path', type=str,
                    default="data/bootstrap_reranker/bart_queries/", help="Path to save retrieved and filtered instances")

    return parser


def main():
    args = get_parser().parse_args()
    log_args(args)

    # Load customized grouping
    group_of = load_grouping()
    make_dir(args.output_path)

    # Load distant supervision data
    # Format is (query, positive, negative) tuples 
    ds_data = load_data_from_json(DS_DATA_PATH)

    queries = defaultdict(list) # [group] -> list of queries

    # Loop through all the tuples in ds data
    # Build a map of group -> [querie examples]
    for tp in ds_data:
        query = tp['query_examples']
        # All items from query are from the same group (that's how we generated ds data)
        dataset, _, _, _ = query[0][2].split("|")
        group = group_of[dataset]

        # Populate the map
        queries[group].append(query)
    
    selected_queries = []
    for i in range(args.N):
        # Randomly pick a group
        g = random.choice(list(queries.keys()))
        # Randomly pick a query in that group
        query = random.choice(queries[g])
        selected_queries.append(query)

    save_json(args.output_path, f"bart_queries_{len(selected_queries)}.json", selected_queries)  
    log(f"Seleceted {len(selected_queries)} queries. Done!")

if __name__ == "__main__":
    main()
