# logs = ["logs/bart0-zeroshot_Random/RET_Random-unsupervised_no-1-16-512-{prefix[0]}-{task_name}-{prefix[1]}-2.log",
#         "logs/bart0-zeroshot_SentenceTransformer/RET_SentenceTransformer-unsupervised_no-1-16-512-{prefix[0]}-{task_name}-{prefix[1]}-2.log",
#         "logs/bart0-zeroshot_BART/RET_BART-unsupervised_no-1-16-512-{prefix[0]}-{task_name}-{prefix[1]}-2.log",
#         "logs/bart0-zeroshot_Random/RET_Random-unsupervised_rerank-2-16-512-{prefix[0]}-{task_name}-{prefix[1]}-2.log",
#         "logs/bart0-zeroshot_SentenceTransformer/RET_SentenceTransformer-unsupervised_rerank-2-16-512-{prefix[0]}-{task_name}-{prefix[1]}-2.log",
#         "logs/bart0-zeroshot_BART/RET_BART-unsupervised_rerank-1.5-16-512-{prefix[0]}-{task_name}-{prefix[1]}-2.log",
#         ]

import json
import altair as alt
import numpy as np
import pandas as pd

with open("metax/distant_supervision/task_groupings.json") as f:
    task_grouping = json.load(f)

def get_group_name(task_name):
    for group_name in task_grouping:
        if task_name in task_grouping[group_name]:
            return group_name
    return None

data_paths = {
   "Random": "retrieved_data/zs_Random/RET_Random-unsupervised_no-1-16-512-42-{task_name}-6e-6-2_seed={seed}_round#{round_id}.json",
   "Random_Rerank": "retrieved_data/zs_Random_reranked/RET_Random-unsupervised_rerank-2-16-512-42-{task_name}-6e-6-2_seed={seed}_round#{round_id}.json",
}

task_names = [
    "piqa",
    "ai2_arc-ARC-Challenge",
    "squad_v2",
    "openbookqa-main",
    "super_glue-wic",
    "super_glue-wsc.fixed",
    "winogrande-winogrande_xl",
    "super_glue-cb",
    "anli_r3",
    "hellaswag",
]
round_ids = [0,1,2,3,4]

# method_name = "Random"
method_name = "Random_Rerank"

all_group_names = sorted(list(task_grouping.keys()))

all_data = {"U":[], "G":[], "p": []}
for task_name in task_names:
    ret_data = []
    for round_id in round_ids:
        path = data_paths[method_name].replace("{task_name}", task_name).replace("{round_id}", str(round_id))
        if method_name.startswith("Random"):
            path = path.replace("{seed}", str(42+round_id))
        else:
            path = path.replace("{seed}", "42")
        # print(path)
        with open(path) as f:
            ret_data += json.load(f)
        # print(len(ret_data))
    ret_task_names = [item[2].split("|")[0] for item in ret_data]
    group_names = [get_group_name(t) for t in ret_task_names]
    # print(ret_task_names)
    # print(group_names)
    distribution = {}
    for g in all_group_names:
        # distribution[g] = 
        percent = group_names.count(g)/len(group_names)
        all_data["U"].append(task_name)
        all_data["G"].append(g)
        all_data["p"].append(percent)

    # print(distribution)


# print(all_group_names)
 

df = pd.DataFrame(all_data)

import seaborn as sb

import matplotlib.pyplot as plt




# fig = alt.Chart(df).mark_rect().encode(
#     x='U:O',
#     y='G:O',
#     color='p:Q'
# )

# fig = fig.properties(width=1000, height=1000).configure_axis(
#     labelFontSize=18,
#     titleFontSize=16, 
# ).configure_legend(titleFontSize=0, labelFontSize=15, orient='bottom-right', strokeColor='gray',
#     fillColor='#EEEEEE',
#     padding=10,
#     cornerRadius=5,).configure_title(
#     fontSize=20,
#     font='Courier',
#     anchor='middle',
#     orient="top", align="center",
#     color='black'
# )

# fig.save('test.png')
