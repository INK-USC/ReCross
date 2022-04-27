import os
import json
import random

from metax.config import DOWNSTREAM_DATASETS

# Dataset directory with prompted dataset files
DATA_ROOT='data/'
# datasets = UPSTREAM_DATASETS 

datasets = [
"super_glue-wsc.fixed",
"winogrande-winogrande_xl",
"super_glue-cb",
"super_glue-rte",
"anli_r1",
"anli_r2",
"anli_r3",
"super_glue-copa",
"story_cloze-2016",
"hellaswag",
"super_glue-wic",
] 
concat_all_data = []
for dataset in datasets:
    print(f"Loading {dataset}")
    files = os.listdir(f"{DATA_ROOT}/{dataset}")
    
    # Normal processing
    train_data = []
    if 'test-1000.json' in files:
        with open(f"{DATA_ROOT}/{dataset}/test-1000.json") as f:
            train_data += json.load(f)
    concat_all_data += train_data
print(f"len(concat_all_data)={len(concat_all_data)}")


all_data = []
for item in concat_all_data:
    all_data.append({"input_text": item[0], "output_text": item[1][0]})

random.shuffle(all_data) 

with open("data/downstream_test.json", "w") as f:
    for item in all_data:
        f.write(json.dumps(item) + "\n")
 