"""
    Generate distant supervision data from upstream data
    
    Usage
    -----------
    python metax/distant_supervision/ds_gen.py \
        --eval_reserve task \
        --N_eval 3000 \
        --N_train 30000 \
        --n_query 8  --n_positive 8 --n_negative 8 
"""
import json
import argparse
import random
from copy import deepcopy
from collections import defaultdict
from functools import reduce
from operator import add
from math import ceil
from tokenize import group
from tqdm import tqdm

from metax.distant_supervision.hidden_tasks import TEST_TASKS


UPSTREAM_DATA_PATH = "data/bart0_upstream_train_lines.json"
OUTPUT_FILE_PATH = "data/"
GROUPING_PATH = "metax/distant_supervision/task_groupings.json"
N_raw = 0
# A set of all the dataset names in upstream training data.
dataset_names_ = set()


def get_all_dataset_names(data):
    """
        Extract all dataset names from raw data.
    """
    names = set()
    for dp in data:
        dataset, _, _, _ = dp[2].split("|")
        names.add(dataset)
    return names


def load_upstream_data(upstream_data_path=UPSTREAM_DATA_PATH):
    """
        Load raw upstream data from a json file. 
    """
    print(f"Loading data from {upstream_data_path}")
    data = json.load(open(upstream_data_path))
    global N_raw, dataset_names_
    N_raw = len(data)
    dataset_names_ = get_all_dataset_names(data)
    return data


def load_grouping(grouping_path=GROUPING_PATH):
    """
        Load customized grouping from a json file. 

        Returns
        -------
        ds_to_group : a map from dataset name to group name
    """
    print(f"Loading customized grouping from {grouping_path}")
    input_grouping = json.load(open(grouping_path))

    # Change format to dataset_name:grouping
    ds_to_group = {}
    for name, datasets in input_grouping.items():
        for d in datasets:
            ds_to_group[d] = name
    return ds_to_group


def group_data(data, grouping, valid_groups=None, valid_tasks=None):
    """
        Helper function used by group raw data.
        Not intended to be called by main directly. 

        Parameters
        ----------
        data : raw upstream data
        grouping : a map from dataset name to group name
        valid_groups/valid_datasets : filters for groups/datasets, no filter if None
    """
    grouped = defaultdict(lambda: defaultdict(list))
    for dp in data:
        dataset, _, tplt, _ = dp[2].split("|")

        if valid_groups and grouping[dataset] not in valid_groups:
            continue
        if valid_tasks and dataset not in valid_tasks:
            continue

        grouped[grouping[dataset]][dataset].append(dp)
    print("Finshed generating grouping...")
    return grouped


def group_raw_data(data, grouping=None, reserve_by="group"):
    """
        Index raw data with a 2-level dictionary

        Returns
        --------
        map[grouping][dataset] -> list of upstream instances
    """
    # Note that the terms 'task' and 'dataset' have the same meaning in the following context
    # They both mean the name of a dataset-subset pair in promptsource NLP data collection.
    print("Generating grouping...")
    all_tasks = dataset_names_
    all_groups = set(grouping.values())

    if reserve_by == 'group':
        eval_groups = set(random.sample(all_groups, int(len(all_groups)*0.3)))
        train_groups = all_groups - eval_groups
        print(f"Groups for training dataset {train_groups}")
        print(f"Groups for evaluation dataset {eval_groups}")
        grouped_training = group_data(
            data, grouping, valid_groups=train_groups)
        grouped_eval = group_data(data, grouping, valid_groups=eval_groups)
    elif reserve_by == 'task':
        eval_tasks = TEST_TASKS
        train_tasks = all_tasks - eval_tasks
        print(f"Tasks for training dataset {train_tasks}")
        print(f"Tasks for evaluation dataset {eval_tasks}")
        grouped_training = group_data(data, grouping, valid_tasks=train_tasks)
        grouped_eval = group_data(data, grouping, valid_tasks=eval_tasks)
    else:
        assert False, f"Unknown test set reserve setting {reserve_by}"
    return grouped_training, grouped_eval


def get_parser():
    parser = argparse.ArgumentParser(
        description="Generate Distant Supervision Data")
    parser.add_argument("--N_eval", type=int, default=100,
                        help="How many ds instances to generate for eval.")
    parser.add_argument("--N_train", type=int, default=10000,
                        help="How many train instances to generate for train.")
    parser.add_argument("--eval_reserve", type=str, default="group", choices=[
                        'group', 'task'], help="Reserve some group/task for testing dataset.")
    parser.add_argument("--n_query", type=int, default=16,
                        help="Length of query list in each instance.")
    parser.add_argument("--n_positive", type=int, default=16,
                        help="Length of positive list in each instance.")
    parser.add_argument("--n_negative", type=int, default=16,
                        help="Length of negative list in each instance.")
    return parser


def gen_N(grouped, N, args):
    """
        Generate N distant supervision instances from grouped

        Returns
        -------
        results : a list of dicts each has query, positive, and negative examples.
    """
    # list of tuple (group, size), sorted by group size
    # Maybe generating queries from small groups can increase the total number of instances generated

    groups = list(grouped.keys())
    tasks_of = {}
    for g in groups:
        tasks_of[g] = list(grouped[g].keys())

    results = []
    for _ in tqdm(range(N)):
        query, positive, negative = [], [], []

        curr_group = random.choice(groups)
        que_task = random.choice(tasks_of[curr_group])
        rem_tasks = [t for t in tasks_of[curr_group] if t != que_task]
        rem_groups = [g for g in groups if g != curr_group]

        for _ in range(args.n_query):
            query.append(random.choice(grouped[curr_group][que_task]))

        for _ in range(args.n_positive):
            positive.append(random.choice(
                grouped[curr_group][random.choice(rem_tasks)]))

        for _ in range(args.n_negative):
            rand_g = random.choice(rem_groups)
            rand_t = random.choice(tasks_of[rand_g])
            negative.append(random.choice(grouped[rand_g][rand_t]))

        results.append({"query_examples": query,
                        "positive_examples": positive,
                        "negative_examples": negative})

    return results


def main():
    args = get_parser().parse_args()
    random.seed(7)

    grouped_train, grouped_eval = group_raw_data(
        load_upstream_data(), grouping=load_grouping(), reserve_by=args.eval_reserve)

    # Generate Eval and Train datasets separately
    results_train = gen_N(grouped_train, args.N_train, args)
    results_eval = gen_N(grouped_eval, args.N_eval, args)

    print(
        f"Generated {len(results_train)} distant supervision instances as training dataset")
    print(
        f"Generated {len(results_eval)} distant supervision instances as evaluation dataset")

    json.dump(results_train[:-3000], open(OUTPUT_FILE_PATH +
              "ds_from_bart0_upstream_train.json", "w"), indent=4)
    json.dump(results_train[-3000:], open(OUTPUT_FILE_PATH +
              "ds_from_bart0_upstream_dev.json", "w"), indent=4)
    json.dump(results_eval, open(OUTPUT_FILE_PATH +
              "ds_from_bart0_upstream_test.json", "w"), indent=4)


if __name__ == "__main__":
    main()
