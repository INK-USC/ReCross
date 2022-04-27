"""
    WARNING: this version of ds_gen is deprecated, we now use a simpler version of ds_gen

    Generate distant supervision data from upstream data
Usage:

python metax/distant_supervision/ds_gen.py \
    --customized_grouping \
    --eval_reserve group \
    --n_query 16  --n_positive 16 --n_negative 16 
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



UPSTREAM_DATA_PATH = "data/bart0_upstream_train_lines.json"
OUTPUT_FILE_PATH = "data/"
GROUPING_PATH = "metax/distant_supervision/task_groupings.json"
N_raw = 0
# A set of all the dataset names in upstream training data.
dataset_names_ = set()


def load_upstream_data():
    print(f"Loading data from {UPSTREAM_DATA_PATH}")
    data = json.load(open(UPSTREAM_DATA_PATH))
    global N_raw, dataset_names_
    N_raw = len(data)
    dataset_names_ = get_all_dataset_names(data)
    return data

def load_grouping():
    print(f"Loading customized grouping from {GROUPING_PATH}")
    input_grouping = json.load(open(GROUPING_PATH))

    # Change format to dataset_name:grouping
    ds_to_group = {}
    for name, datasets in input_grouping.items():
        for d in datasets:
            ds_to_group[d] = name
    return ds_to_group

def get_all_dataset_names(data):
    names = set()
    for dp in data:
        dataset,_,_,_ = dp[2].split("|")
        names.add(dataset)
    return names


def group_data(data, grouping, valid_groups, valid_datasets):
    """
        Helper function used by group raw data.
        Not intended to be called by main directly. 

        Parameters:
        -----------
        data: same format as raw data
        grouping: a map from dataset name to group name
        valid_groups/valid_datasets: filters for groups/datasets
    """
    grouped = defaultdict(lambda: defaultdict(list))
    for dp in data:
        dataset,_,tplt,_ = dp[2].split("|")

        if dataset not in valid_datasets: continue
        if grouping[dataset] not in valid_groups: continue
        
        grouped[grouping[dataset]][tplt].append(dp)
    print("Finshed generating grouping...")
    return grouped


def group_raw_data(data, grouping=None, eval_reserve=None):
    """
        Index raw data with a 2-level dictionary
            d[grouping][template] -> list of instances
        When grouping=None use dataset name as grouping
    """
    print("Generating grouping...")
    all_tasks = dataset_names_
    if grouping: 
        all_groups = set(grouping.values())
    else:
        all_groups = all_tasks
        grouping = {t:t for t in all_tasks}
    
    # Note that the terms 'task' and 'dataset' have the same meaning in the following context
    # They both mean the name of a dataset-subset pair in promptsource NLP data collection. 
    if eval_reserve == "group":
        eval_groups = set(random.sample(all_groups, int(len(all_groups)*0.3)))
        train_groups = all_groups - eval_groups

        print(f"Groups for evaluation dataset {eval_groups}")
        print(f"Groups for training dataset {train_groups}")

        grouped_training = group_data(data, grouping, valid_groups=train_groups, valid_datasets=all_tasks)
        grouped_eval = group_data(data, grouping, valid_groups=eval_groups, valid_datasets=all_tasks)
        return grouped_training, grouped_eval
    elif eval_reserve == "task":
        eval_tasks = set(random.sample(all_tasks, int(len(all_tasks)*0.2)))
        train_tasks = all_tasks - eval_tasks

        print(f"Tasks for evaluation dataset {eval_tasks}")
        print(f"Tasks for training dataset {train_tasks}")

        grouped_training = group_data(data, grouping, valid_groups=all_groups, valid_datasets=train_tasks)
        grouped_eval = group_data(data, grouping, valid_groups=all_groups, valid_datasets=eval_tasks)
        return grouped_training, grouped_eval
    else:
        grouped = group_data(data, grouping, valid_groups=all_groups, valid_datasets=all_tasks)
        return grouped, None



def get_parser():
    parser = argparse.ArgumentParser(description="Generate Distant Supervision Data")
    parser.add_argument('--customized_grouping', action='store_true', help="Whether to use customized grouping.")
    parser.add_argument("--eval_reserve", type=str, choices=['group', 'task'], help="Reserve some group/task for test dataset.")
    parser.add_argument("--n_query", type=int, default=16, help="Length of query list in each instance.")
    parser.add_argument("--n_positive", type=int, default=16, help="Length of positive list in each instance.")
    parser.add_argument("--n_negative", type=int, default=16, help="Length of negative list in each instance.")
    return parser

def grp_sz(grouped, grp):
    """
        Returns total number of instances within a group (sum over all templates)
    """
    if not len(grouped[grp]):
        return 0
    return reduce(add, [len(grouped[grp][t]) for t in grouped[grp].keys()])  

def tplt_sz(grouped, grp, tplt):
    """
        Returns instances within a group within a template
    """
    return len(grouped[grp][tplt])

def count_group_sizes(grouped):
    """
        Returns a map of:
            group_name : total number of instances in the group
    """
    count = defaultdict(int)
    for grp in list(grouped.keys()):
        sz = grp_sz(grouped, grp)
        count[grp] = sz
    return count

def count_template_sizes(grouped, grp):
    """
        Return a map of:
            template_name : total number of instances in the group with the template
    """
    count = defaultdict(int)
    for t in list(grouped[grp].keys()):
        sz = tplt_sz(grouped, grp, t)
        if sz > 0:
            count[t] = sz
    return count

def distribute_templates(tplt_and_sz, args):
    """
        Decide how to distribute templates for 'query' and 'positive'
        This algorithm has two goals:
            Make sure there's no template overlap between query and positive.
            Maximize the variety of template types within query and positive respectively. 

        This version of implementation just randomly divide the templates types in two and try
        to assign instances. We might be able to squeeze out more distant supervision instances if we
        optimize this part. Finding the theoretically optimal implementation is not a trivial algorithm problem. 
    """
    n_pos = args.n_positive
    n_que = args.n_query
    n_t = len(tplt_and_sz)
    left = n_t // 2
    
    left_ans = defaultdict()
    right_ans = defaultdict()

    # If we shuffle 10 times and still can't get a good partition then give up.
    trials = 10
    while trials > 0:
        random.shuffle(tplt_and_sz)
        lsz = reduce(add, [e[1] for e in tplt_and_sz[0:left]])
        rsz = reduce(add, [e[1] for e in tplt_and_sz[left:]])
        if lsz >= n_que and rsz >= n_pos: 
            lrem = n_pos
            rrem = n_que
            for i in range(left):
                portion = min(ceil(n_que*tplt_and_sz[i][1]/lsz), lrem)
                left_ans[tplt_and_sz[i][0]] = portion
                lrem -= portion
            for i in range(left, n_t):
                portion = min(ceil(n_pos*tplt_and_sz[i][1]/rsz), rrem)
                right_ans[tplt_and_sz[i][0]] = portion
                rrem -= portion

            # TODO: delete this 
            # Sanity Check
            lsz = reduce(add, [e[1] for e in left_ans.items()])
            rsz = reduce(add, [e[1] for e in right_ans.items()])
            assert lsz == n_que and rsz == n_pos, "Distribution function error!"

            return left_ans, right_ans
        trials -= 1
    return None, None

def select_negative(grouped, curr_grp, N):
    """
        Randomly selects negative instances from other groups
    """
    negative = []
    while len(negative) < N:
        sweep_empty_groups(grouped)
        groups = list(grouped.keys())
        groups.remove(curr_grp)
        if not len(groups):
            return []
        grp = random.choice(groups)
        if not len(list(grouped[grp].keys())):
            print(grp, tplt)
            print(count_group_sizes(grouped))
        tplt = random.choice(list(grouped[grp].keys()))
        negative.append(grouped[grp][tplt][0])
        grouped[grp][tplt].pop(0)
        if not len(grouped[grp][tplt]):
            del grouped[grp][tplt]
        sweep_empty_groups(grouped)

    return negative

def sweep_empty_groups(grouped):
    """
        Clean the map. Delete the groups that are already empty. 
    """
    grps_sz = count_group_sizes(grouped)
    to_delete = [grp for grp,sz in grps_sz.items() if sz == 0]
    for g in to_delete:
        del grouped[g]

def gen_all(grouped, args):
    """
        Generate all distant training instance. 
        This function can be used by both task level and group level
        
        Note that we generate all ds instances then randomly select N of them. 
        This way we avoid having overlaps among ds instances. 
    """
    # list of tuple (group, size), sorted by group size
    # Maybe generating queries from small groups can increase the total number of instances generated


    sweep_empty_groups(grouped)
    grp_ascend_sz = sorted(count_group_sizes(grouped).items(),key=lambda kv: kv[1])
    n_que = args.n_query
    n_pos = args.n_positive
    n_neg = args.n_negative
    results = []

    for grp, sz in grp_ascend_sz:
        print(f"{grp}:{sz}")
    pbar = tqdm(total=N_raw//(args.n_query + args.n_negative + args.n_positive)) # Upperbound
    pbar.set_description("ds instance generated")

    # Generation
    for grp, sz in grp_ascend_sz:
        sz = grp_sz(grouped, grp)       
        # While this group still has enough instances
        # And there are at least 2 non-empty templates
        while sz >= n_que + n_pos and len(count_template_sizes(grouped, grp)) > 1:  
            # Go through templates
            sweep_empty_groups(grouped)
            tplt_and_sz = list(count_template_sizes(grouped, grp).items())
            que_quota, pos_quota = distribute_templates(tplt_and_sz, args)
            if not que_quota: break

            query, positive, negative = [],[],select_negative(grouped, grp, n_neg)
            if not len(negative): break
            
            for tplt, sz in que_quota.items():
                query += grouped[grp][tplt][:sz]
                grouped[grp][tplt] = grouped[grp][tplt][sz:]
                if not len(grouped[grp][tplt]):
                    del grouped[grp][tplt]

            for tplt,sz in pos_quota.items():
                positive += grouped[grp][tplt][:sz]
                grouped[grp][tplt] = grouped[grp][tplt][sz:]
                if not len(grouped[grp][tplt]):
                    del grouped[grp][tplt]


            results.append({"query_examples": query, 
                            "positive_examples": positive, 
                            "negative_examples": negative})
            
            pbar.update(1)

            sz = grp_sz(grouped, grp) 

    return results


def main():
    args = get_parser().parse_args()
    random.seed(7)
    # Option 1
    if args.customized_grouping:
        print("Using customized grouping.")
        grouped = group_raw_data(load_upstream_data(),grouping=load_grouping(), eval_reserve=args.eval_reserve)
    else:
        print("Using default grouping by dataset.")
        grouped = group_raw_data(load_upstream_data(), eval_reserve=args.eval_reserve)

    grouped_train, grouped_eval = grouped
    if args.eval_reserve:
        # Generate Eval and Train datasets separately 
        results_train = gen_all(grouped_train, args)
        results_eval = gen_all(grouped_eval, args)
        print(f"Generated {len(results_train)} distant supervision instances as training dataset")
        print(f"Generated {len(results_eval)} distant supervision instances as evaluation dataset")
        json.dump(results_train, open(OUTPUT_FILE_PATH + "ds_from_bart0_upstream_train.json", "w"), indent=4)
        json.dump(results_eval, open(OUTPUT_FILE_PATH + "ds_from_bart0_upstream_eval.json", "w"), indent=4)
    else:
        # This branch is deprecated (allows task or group level overlap between train set and eval set). 
        results = gen_all(grouped_train, args)
        print(f"Generated {len(results)} distant supervision instances")
        json.dump(results, open(OUTPUT_FILE_PATH + "ds_from_bart0_upstream.json", "w"), indent=4)

if __name__ == "__main__":
    print("WARNING: this version of ds_gen is deprecated, we now use a simpler version of ds_gen")