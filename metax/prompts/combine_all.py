import os
import json
import random


from metax.config import UPSTREAM_DATASETS, CSR_UPSTEREAM_TASKS


random.seed(2022)

# Dataset directory with prompted dataset files
# DATA_ROOT='data/'
# datasets = UPSTREAM_DATASETS 

# with open("data/split.json") as f:
#     data_splits = json.load(f)

# datasets = data_splits["T0"]

DATA_ROOT='data_csr/'
datasets = CSR_UPSTEREAM_TASKS 

concat_all_data = []
for dataset in datasets:
    print(f"Loading {dataset}")
    files = os.listdir(f"{DATA_ROOT}/{dataset}")
    
    # Normal processing
    train_data = []
    if 'raw-train.json' in files:
        with open(f"{DATA_ROOT}/{dataset}/raw-train.json") as f:
            train_data += json.load(f)
    if 'raw-validation.json' in files:
        with open(f"{DATA_ROOT}/{dataset}/raw-validation.json") as f:
            train_data += json.load(f)
    random.shuffle(train_data)
    concat_all_data += train_data[:50000]   # 50k for the BART0 and 60k for the BART0
print(f"len(concat_all_data)={len(concat_all_data)}")


all_data = []
for item in concat_all_data:
    # all_data.append(item)
    all_data.append({"input_text": item[0], "output_text": item[1][0], "example_id": item[2]})

random.shuffle(all_data)
upstream_train = all_data[:-10000]
upstream_dev = all_data[-10000:]



with open("data/CSR_upstream_train.json", "w") as f:
    for item in upstream_train:
        f.write(json.dumps(item) + "\n")

with open("data/CSR_upstream_dev.json", "w") as f:
    for item in upstream_dev:
        f.write(json.dumps(item) + "\n")
    