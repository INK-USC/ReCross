"""
This file load datasets from datasets lib, process them with prompt templates and save them as local files.

python metax/prompts/prompt.py -1

for i in {0..21}
do
    python metax/prompts/prompt.py $i &
    # 1>/dev/null 2>&1 &
done


"""

import os
import json
import random
from tqdm import tqdm
from datasets import load_dataset
from promptsource.templates import TemplateCollection
import sys
from IPython import embed
# Change this to set the directory to save data
# data_root = '/home/beiwen/MetaCross/data'
data_root = 'data_csr/' # ln -s /path/to/data  ./data  #
os.makedirs(data_root, exist_ok=True)

random.seed(42)

# Get all the prompts
from metax.config import UPSTREAM_DATASETS, DOWNSTREAM_DATASETS, CSR_UPSTEREAM_TASKS

collection = TemplateCollection()

all_names = CSR_UPSTEREAM_TASKS
print(f"{len(all_names)} datasets available")
names = [i.split('-', maxsplit=1) for i in all_names]

index = sys.argv[1]
if index and int(index) >= 0:
    names = [names[int(index)]]
else:
    print(names)
    exit()

for name in names:
    print(f"Loading dataset {name}")

    name_str = "-".join(name)

    # Load datasets from datasets lib
    if name[0] == "paws_x":
        name[0] = "paws-x"
    
    if name[0] == "story_cloze":
        datasets = load_dataset("story_cloze", "2016", "data/story_cloze_local")
    elif name[0] == "cnn_dailymail":
        datasets = load_dataset(*name, ignore_verifications=True) # , download_mode="force_redownload"
    else:
        datasets = load_dataset(*name)

    if int(sys.argv[1]) < 0:
        # skip the processing first (only downloading the datasets)
        del datasets
        continue

    # Process the datasets
    if name[0] == 'anli':
        splits = ['train_r1', 'train_r2', 'train_r3', 'dev_r1', 'dev_r2', 'dev_r3']
    else:
        splits = ['train', 'validation']
    for split in splits:

        num_template_per_example = 1

        samples = []
        dataset = datasets[split]
        prompts = collection.get_dataset(*name)
        if len(dataset) >= 100000:
            print(f"\tcut to 100k for ${name}")
            dataset = dataset.select(range(100000))
        
        if len(dataset) <= 5000:
            num_template_per_example = 10
        elif len(dataset) <= 10000:
            num_template_per_example = 7
        elif len(dataset) <= 50000:
            num_template_per_example = 5

        if name[0] in ["hotpotqa"]:
            num_template_per_example = 1

        if len(prompts.all_template_names) < num_template_per_example:
            num_template_per_example = len(prompts.all_template_names)
    
        

        print(f"\tnum_template_per_example={num_template_per_example}")
        for idx, item in tqdm(enumerate(dataset),
                              total=len(dataset),
                              desc=f"Processing {name}--Split {split}"):
            applied = []
            
            while len(applied) < num_template_per_example:
                while True:
                    # randomly pick one template and apply to the data sample
                    template_idx = random.randint(
                        0,
                        len(prompts.all_template_names) - 1)
                    if template_idx in applied:
                        continue

                    template_name = prompts.all_template_names[template_idx]
                    prompt = prompts[template_name]
                    r = prompt.apply(item)

                    # Continue only when reasonable results are generated
                    if len(r) > 1:
                        applied.append(template_idx)
                        break

                sample = [r[0], [r[1]], f'{name_str}|{split}|{template_name}|#{idx}']
                samples.append(sample)

        # Save the processed samples
        os.makedirs(f'{data_root}/{name_str}', exist_ok=True)
        with open(f'{data_root}/{name_str}/raw-{split}.json', 'w') as f:
            json.dump(samples, f, indent='    ')
