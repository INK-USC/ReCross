""" 
python metax/prompts/divide.py

"""

import os
import json
import random
from random import shuffle

from metax.config import UPSTREAM_DATASETS, DOWNSTREAM_DATASETS, CSR_TARGET_TASKS, CSR_UPSTEREAM_TASKS

# Dataset directory with prompted dataset files
DATA_ROOT='data_csr'
# datasets = UPSTREAM_DATASETS + DOWNSTREAM_DATASETS

datasets = CSR_UPSTEREAM_TASKS 

# datasets = [
#     "ai2_arc-ARC-Easy",
#     "ai2_arc-ARC-Challenge",
#     "piqa",
#     "squad_v2",
#     "hellaswag",
#     "openbookqa-main",
#     "super_glue-multirc",
#     "super_glue-boolq",
#     "super_glue-wic",
#     "super_glue-wsc.fixed",
#     "winogrande-winogrande_xl",
#     "super_glue-cb",
#     "super_glue-rte",
#     "anli"
# ]

# datasets = ["squad_v2"]


random.seed(42)

# Load metrics info used by each dataset/subset/template
metrics_to_use = json.load(open(f"{DATA_ROOT}/supported_metrics.json"))

def suitable_for_EM(data):
    """
        Only keep the instances that can be evaluated by EM
    """
    result = []
    for dp in data:
        dpid = dp[2]
        dataset = dpid.split("|")[0]
        template = dpid.split("|")[2]
        # We only want the instances that can be evaluated by "Accuracy", a.k.a. EM
        if "Accuracy" in metrics_to_use[dataset][template] or "Squad" in metrics_to_use[dataset][template]:
            result.append(dp)
    print(f"Suitable for EM: {len(result)} out of {len(data)}")
    return result

for dataset in datasets:
    print(f"Processing {dataset}")
    files = os.listdir(f"{DATA_ROOT}/{dataset}")
    fewshot_data = {16:{}, 32:{}, 64:{}, 128:{}, 256:{}, 512:{}}

    # Special handling for story_cloze
    if dataset == 'story_cloze-2016':
        with open(f"{DATA_ROOT}/{dataset}/raw-validation.json") as f:
            validation_data = suitable_for_EM(json.load(f))
        with open(f"{DATA_ROOT}/{dataset}/test-1000.json", 'w') as f:
            json.dump(validation_data[:1000], f, indent='    ')
        
        train_data = validation_data[1000:]

        
        for round in range(20):
            shuffle(train_data)
            for shot in list(sorted(fewshot_data.keys())):
                fewshot_data[shot][f"round#{round}"] = train_data[:shot]

        with open(f"{DATA_ROOT}/story_cloze-2016/fewshot.json", 'w') as f:
            json.dump(fewshot_data, f, indent='    ')
        
        continue 
        
    # Special handling for anli
    if dataset == 'anli':
        for i in range(1,4):
            train_data = []
            validation_data = []
            with open(f"{DATA_ROOT}/anli/raw-train_r{i}.json") as f:
                train_data += suitable_for_EM(json.load(f))
            with open(f"{DATA_ROOT}/anli/raw-dev_r{i}.json") as f:
                validation_data += suitable_for_EM(json.load(f))
            
            if len(train_data) < 11000:
                print(f"\tInsufficient train data: {len(train_data)}")
            
            upstream_count = min(10000, len(train_data) - 1000)
            shuffle(train_data)
            # apply the filter on train_data and validation_data
            os.makedirs(f"{DATA_ROOT}/anli_r{i}", exist_ok=True)
            with open(f"{DATA_ROOT}/anli_r{i}/upstream-10000.json", 'w') as f:
                json.dump(train_data[:upstream_count], f, indent='    ')
            with open(f"{DATA_ROOT}/anli_r{i}/validation-1000.json", 'w') as f:
                json.dump(train_data[-1000:], f, indent='    ')
            with open(f"{DATA_ROOT}/anli_r{i}/test-1000.json", 'w') as f:
                json.dump(validation_data[:1000], f, indent='    ')
             
            for round in range(20):
                shuffle(train_data)
                for shot in list(sorted(fewshot_data.keys())):
                    fewshot_data[shot][f"round#{round}"] = train_data[:shot]

            with open(f"{DATA_ROOT}/anli_r{i}/fewshot.json", 'w') as f:
                json.dump(fewshot_data, f, indent='    ')
        continue

    # Normal processing
    if 'raw-train.json' in files:
        if 'raw-validation.json' in files:
            with open(f"{DATA_ROOT}/{dataset}/raw-train.json") as f:
                train_data = suitable_for_EM(json.load(f))
            with open(f"{DATA_ROOT}/{dataset}/raw-validation.json") as f:
                validation_data = suitable_for_EM(json.load(f))
            if len(train_data) < 11000:
                print(f"\tInsufficient train data: {len(train_data)}")

            upstream_count = min(10000, len(train_data) - 1000)
            shuffle(train_data)
            # apply filter on both train_data and validation_data

            with open(f"{DATA_ROOT}/{dataset}/upstream-10000.json", 'w') as f:
                json.dump(train_data[:upstream_count], f, indent='    ')
            with open(f"{DATA_ROOT}/{dataset}/validation-1000.json", 'w') as f:
                json.dump(train_data[-1000:], f, indent='    ')
            with open(f"{DATA_ROOT}/{dataset}/test-1000.json", 'w') as f:
                json.dump(validation_data[:1000], f, indent='    ')
        else:
            with open(f"{DATA_ROOT}/{dataset}/raw-train.json") as f:
                train_data = suitable_for_EM(json.load(f))
            if len(train_data) < 7000:
                print(f"\tInsufficient train data: {len(train_data)}")

            upstream_count = min(10000, len(train_data) - 2000)
            shuffle(train_data)
            # apply the filter on train_data
            with open(f"{DATA_ROOT}/{dataset}/upstream-10000.json", 'w') as f:
                json.dump(train_data[:upstream_count], f, indent='    ')
            with open(f"{DATA_ROOT}/{dataset}/validation-1000.json", 'w') as f:
                json.dump(train_data[-2000:-1000], f, indent='    ')
            with open(f"{DATA_ROOT}/{dataset}/test-1000.json", 'w') as f:
                json.dump(train_data[-1000:], f, indent='    ')

        for round in range(20):
            shuffle(train_data)
            for shot in list(sorted(fewshot_data.keys())):
                fewshot_data[shot][f"round#{round}"] = train_data[:shot]

        with open(f"{DATA_ROOT}/{dataset}/fewshot.json", 'w') as f:
            json.dump(fewshot_data, f, indent='    ')
    else:
        print(f"=> Error: {dataset} raw-train.json does not exist")
        continue