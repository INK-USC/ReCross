import logging
import json
from os import walk
from tqdm import tqdm
from itertools import product
import pandas as pd
import os

def init_logger(description="Main"):
    main_logger = logging.getLogger("Main")
    formatter = logging.Formatter("[%(asctime)s][%(levelname)s\t] %(message)s",
                                  datefmt='%Y-%m-%d %H:%M:%S')
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    main_logger.addHandler(ch)
    main_logger.setLevel(logging.INFO)
    return main_logger

def load_upstream_dataset_list(model_type="T0"):
    return load_data_from_json("data/split.json")[model_type]    


def load_data_from_json(path):
    """
        Load raw data from a json file. 
    """
    data = json.load(open(path))
    return data

def make_dir(path):
    """
        Make a dir if not already exists
    """
    if not os.path.exists(path):
        os.makedirs(path)


def save_json(path, fname, data):
    if path[-1] != "/":
        path += "/"
    make_dir(path)
    json.dump(data, open(path+fname, 'w'), indent=4)

def get_folders(path):
    return next(walk(path), (None, None, []))[1]


def get_files(path):
    return next(walk(path), (None, None, []))[2]

def get_task_from_id(id):
    task,_,_,_ = id.split("|")
    return task


def assemble_pairs(rdata, cutoff=-1, mode="by_row"):
    all_data = []
    if cutoff > 0:
        rdata = random.sample(rdata, cutoff)
    print(f"len(rdata)={len(rdata)}")
    def gen_one(sa, sb, label):
        """
            Generate one reranker training data instance
            
            Parameters:
            ----------
                sa: task instance a
                sb: task instance b
                label: 0 or 1, whether the two are of the same type. 
        """
        return [sa[0], sb[0], label]

    if mode == "by_row":
        # 2 pairs for each row, 1 positive, 1 negative
        for tp in tqdm(rdata):
            # Iterate each 3-tuple
            que = tp['query_examples']
            pos = tp['positive_examples']
            neg = tp['negative_examples']
            sz = min(len(que), len(pos), len(neg))
            for i in range(sz):
                # Create 1 positive entry and 1 negative entry
                all_data.append(gen_one(que[i],pos[i],1))
                all_data.append(gen_one(que[i],neg[i],0))
    elif mode == "product":
        # Cartesian product
        for tp in tqdm(rdata):
            # Iterate each 3-tuple
            que = tp['query_examples']
            pos = tp['positive_examples']
            neg = tp['negative_examples']

            # Use cartesian product
            positive_pairs = [gen_one(q, p, 1) for q,p in product(que,pos)]
            negative_pairs = [gen_one(q, n, 0) for q,n in product(que,neg)]

            all_data += positive_pairs
            all_data += negative_pairs
    return all_data

def create_model_dataframe(data):
    """
        Convert data to the dataformat required by
        classification model of simpletransformers
    """
    model_df = pd.DataFrame(data)
    model_df.columns = ["text_a", "text_b", "labels"]
    return model_df
