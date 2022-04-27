"""
    given a set of query, retrieve using bart retriever
"""
import argparse
from dataclasses import dataclass
import logging
import random
from tokenize import group
from typing import List
import numpy as np
import faiss

from metax.meta_algs.bartbase_index import BartIndexManager
from metax.reranker_bootstrap.utils import *

# Task grouping meta-data
from metax.distant_supervision.ds_gen import GROUPING_PATH, load_grouping


MODLE_TYPE = "yuchenlin/BART0"
main_logger = init_logger()

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

    parser.add_argument('--query_path', type=str,
                        help="Path to query json file (a list of queries)")
    parser.add_argument('--N', type=int, default=60,
                        help="How many candidate instances to select from retrieved instances for each instance in a query")
    parser.add_argument('--output_path', type=str,
                        help="Path to save retrieved and filtered <query, retrieved> tuples")
    parser.add_argument('--shards', type=int, default=5,
                        help="How many shards to separate the final results into. (depends on how many jobs of generate better tuple you want to run in parallel later.)")
    parser.add_argument('--cache_path', type=str,
                        default="memory_cache/bart0_memory.pkl",
                        help="Path to bart0 memory index cache")

    return parser


def get_groups(group_of : map, data : List) -> set:
    """
        parameters:
        -----------
        group_of : a map from task name to group name
        data : list of formatted instances

        return:
        -------
        a set containing groups appeared in this list of formatted instances
    """
    groups = set()
    for dp in data:
        dataset, _, _, _ = dp[2].split("|")
        groups.add(group_of[dataset])
    return groups

def get_tasks(data : List) -> set:
    """
    parameters:
    -----------
    data : list of formatted instances

    return:
    -------
    a set containing tasks appeared in this list of formatted instances
    """
    tasks = set()
    for dp in data:
        task_name, _, _, _ = dp[2].split("|")
        tasks.add(task_name)
    return tasks

def init_bart_index_manger(cache_path):
    # BartIndexManager takes some args from MetaX pipeline's argparse
    # Since we are not using the pipeline args
    # We want to recreate the used ones here
    manager_args = argparse.Namespace(
        n_gpu=1,
        retriever_mode="BART",
        model_type=MODLE_TYPE,
        num_shards_for_indexing=1,
        shard_id_for_indexing=0,
        query_aggregation_mode="aggregate_choices",
        max_input_length=512,
        max_output_length=512,
        do_lowercase=False,
        append_another_bos=True,
        ret_merge_mode="unsupervised",
        predict_batch_size=8
    )
    bart_index_manager = BartIndexManager(main_logger, manager_args)
    bart_index_manager.load_memory_from_path(cache_path)
    return bart_index_manager



def main():
    args = get_parser().parse_args()
    log_args(args)
    make_dir(args.output_path)
    curr_uid = 0 # To index the (queries,retrieved) pairs

    bart_index_manager = init_bart_index_manger(args.cache_path)

    group_of = load_grouping() # Load customized grouping

    queries = load_data_from_json(args.query_path) # Load all queries
    log(f"Loaded {len(queries)} queries")
    all_retrieved = []
    for i,query in enumerate(queries):
        log(f"Start retrieving using {i+1}/{len(queries)} query.")
        query_task_set = get_tasks(query)
        assert len(query_task_set) == 1, f"Found instances from {len(query_task_set)} different tasks!"        
        query_task = query_task_set.pop()


        #we only want tasks not in the same group as query
        from_different_task = []
        # Upper bound for upsampling
        upper_bound = 7e5 // len(query) * len(query)
        upsample_sz = min(40000 * len(query), upper_bound)
        required_sz = args.N * len(query)
        # Keep increasing upsample size if we cannot get enough examples
        while len(from_different_task) < required_sz:
            retrieved = bart_index_manager.retrieve_from_memory(
                query,
                # by design of BartIndexManager, this has to be a multiple of len(query_examples)
                sample_size= upsample_sz,
                mode="input",
                seed=1234)

            tasks_appeared = set()
            for dp in retrieved:
                dataset, _, _, _ = dp[2].split("|")
                if dataset != query_task:
                    from_different_task.append(dp)
                    tasks_appeared.add(dataset)
            

            log(f"{len(from_different_task)} out of {len(retrieved)} instances retrieved by BART retriever are not from the same task as query task ({query_task}).")
            log(f"Tasks from different tasks are {list(tasks_appeared)})")

            if len(from_different_task) < required_sz:
                upsample_sz = min(2*upsample_sz, upper_bound)
                log(f"Cannot get enough examples, increase sample size to {upsample_sz}")
                from_different_task = []




        assert len(from_different_task) >= required_sz, f"Not enough instances to select {len(from_different_task)} < {args.N}. Query group: {query_task}"

        # We want N candidates for each instance in our query
        # Because the list is sorted by similarity, we slice from front
        candidates = from_different_task[:required_sz]

        all_retrieved.append({"query_examples":query, "retrieved_candidates":candidates, "tuple_id":curr_uid})
        curr_uid+=1

    shards = [list(l) for l in np.array_split(all_retrieved, args.shards)]

    for i, shard in enumerate(shards):
        save_json(args.output_path, f"query_retrieved_{i+1}_outof_{len(shards)}.json", shard)

if __name__ == "__main__":
    main()
