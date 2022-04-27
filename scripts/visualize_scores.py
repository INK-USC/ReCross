# %%
import pandas as pd
import numpy as np
import sys
datasets = [
"winogrande-winogrande_xl",
"super_glue-cb",
"super_glue-rte",
"anli_r1",
"anli_r2",
"anli_r3", # "anli",
"story_cloze-2016",
"super_glue-wsc.fixed",
# "super_glue-copa",
"hellaswag",
"super_glue-wic",
]

all_datasets = [
    # "ai2_arc-ARC-Easy",
    # "race-high",
    "piqa",
    "ai2_arc-ARC-Challenge",
    "squad_v2",
    "openbookqa-main",
    # "race-middle",
    # "super_glue-multirc",
    # "super_glue-boolq",
    "super_glue-wic",
    # "super_glue-copa",
    "super_glue-wsc.fixed",
    "winogrande-winogrande_xl",
    "super_glue-cb",
    # "super_glue-rte",
    # "anli_r1",
    # "anli_r2",
    "anli_r3",
    # "story_cloze-2016",
    "hellaswag",
]


bigbench_tasks = [
    # "code_line_description",
    "hindu_knowledge",
    "known_unknowns",
    "logic_grid_puzzle",
    # "misconceptions",
    "movie_dialog_same_or_different",
    # "novel_concepts",
    "strategyqa",
    # "formal_fallacies_syllogisms_negation",
    "vitaminc_fact_verification",
    # "winowhy",
]

csr_downstream_tasks = [
    "squad_v2",
    "super_glue-boolq",
    "super_glue-wic",
    "super_glue-cb",
    "super_glue-rte",
    # "anli_r1",
    # "anli_r2",
    # "anli_r3",
	"glue-mrpc",
	"glue-qqp",
	"rotten_tomatoes",
	"imdb",
	"ag_news"
]


# all_datasets = csr_downstream_tasks



# all_datasets = bigbench_tasks

# %%

def time_judge(line, threshold_time="2022-01-11 00:27:18"):
    if '[2022' in line:
        time_start = line.index("[") +1
        time_end = line.index("]")
        time_str = line[time_start:time_end]
        from dateutil import parser
        time_ = parser.parse(time_str)
        threshold  = parser.parse(threshold_time)
        if time_ < threshold:
            return False
    return True

def get_result(get_log_path_fn, prefix, time_threashold=None, use_soft_score=False):
    results = []
    for dataset in all_datasets:
    # for dataset in datasets:
        logf = get_log_path_fn(prefix, dataset)
        # print(logf)
        try:
            with open(logf) as f:
                lines = f.read().splitlines()
        except Exception as e:
            print(e)
            continue
        seen_column_strs = set()
        for line in lines[::-1]:
            # time = [2022-01-11 17:27:18][INFO	]
            perf_str = "test_perf"

            if "Evaluate" in line and "{'EM':" in line and dataset in line and "round" not in line:
                if time_threashold and not time_judge(line, time_threashold):
                    continue
                # print(line)
                task_ind_start = line.index("Evaluate ") + len("Evaluate ")
                task_ind_end = line.index(" : {'EM':")
                score_ind_start = line.index("{'EM': ") + len("{'EM': ")
                score_ind_end = line.index(",")
                task_name =  line[task_ind_start:task_ind_end].strip()
                assert task_name == dataset

                soft_score_ind_start = line.index("'SoftEM': ") + len("'SoftEM': ")
                soft_score_ind_end = line.index("}")

                score = line[score_ind_start:score_ind_end].strip()
                score = float(score)
                soft_score = line[soft_score_ind_start:soft_score_ind_end].strip()
                soft_score = float(soft_score)

                res = {}
                res["task"] = task_name
                # res["score"] = score
                # res["prefix"] = prefix
                if use_soft_score:
                    res[f"{prefix}"] = soft_score
                else:
                    res[f"{prefix}"] = score

                if prefix in seen_column_strs:
                    continue
                results.append(res)
                seen_column_strs.add(prefix)
            elif perf_str in line and "EM-->" in line and dataset in line:
                if time_threashold and not time_judge(line, time_threashold):
                    continue
                #  [2022-01-24 14:37:00][INFO	] test_perf: ai2_arc-ARC-Easy round #9 with EM--> {'EM': 0.46, 'SoftEM': 0.589}
                res = {}
                task_ind_start = line.index(perf_str+": ") + len(perf_str+": ")
                # task_ind_start = line.index("test_perf: ") + len("test_perf: ")
                task_ind_end = line.index("round")
                round_ind_start = line.index("round #") + len("round #")
                round_ind_end = line.index(" with EM")
                score_ind_start = line.index("{'EM': ") + len("{'EM': ")
                score_ind_end = line.index(",")

                soft_score_ind_start = line.index("'SoftEM': ") + len("'SoftEM': ")
                soft_score_ind_end = line.index("}")
                task_name =  line[task_ind_start:task_ind_end].strip()
                assert task_name == dataset
                round_id = line[round_ind_start:round_ind_end].strip()
                score = line[score_ind_start:score_ind_end].strip()
                score = float(score)
                soft_score = line[soft_score_ind_start:soft_score_ind_end].strip()
                soft_score = float(soft_score)
                res["task"] = task_name
                column_str = f"{'-'.join([str(p) for p in prefix])}@{round_id}"
                column_str = column_str.replace("SentenceTransformer", "SBERT")
                column_str = column_str.replace("two-stage", "2s")
                column_str = column_str.replace("unsupervised", "0s")

                if use_soft_score:
                    res[column_str] = soft_score
                else:
                    res[column_str] = score

                if column_str in seen_column_strs:
                    continue
                results.append(res)
                seen_column_strs.add(column_str)
    return results
# res["average"] = np.mean(list(res.values()))
# print(res)

def process_exp(get_log_path_fn, prefixes, time_threashold=None, use_soft_score=False):
    results = []
    for prefix in prefixes:
        result = get_result(get_log_path_fn, prefix=prefix, time_threashold=time_threashold, use_soft_score=use_soft_score)
        # res["seed"] = str(seed)
        # print(f"{prefix} = {results}")
        results += result
    return results

def clear_and_print(all_results):
    all_results_pd = pd.DataFrame(all_results).sort_values(by=['task']).drop_duplicates().groupby("task").sum().reset_index()

    pd.options.display.float_format = '{:.2%}'.format

    # print(all_results_pd)
    print(all_results_pd.to_csv(index=False))


use_soft_score = True
print(f"use_soft_score={use_soft_score}")
all_results = []
# For Zeroshot Evalution
get_log_path = lambda model_name, task_name: f"logs/{model_name}-zeroshot/{model_name}-zs-{task_name}.log"
all_results += process_exp(get_log_path, ["BART0", "T0_3B"], time_threashold="2022-01-25 00:27:18", use_soft_score=use_soft_score) # ,  "T0_3B"

clear_and_print(all_results)
all_results = []
 






# ----------- # get_log_path = lambda prefix, task_name: f"logs/bart0-zeroshot_Random/RET_Random-unsupervised_rerank-2-32-512-{prefix[0]}-{task_name}-{prefix[1]}-2.log"  
# ----------- # all_results +=  process_exp(get_log_path, [(x, "6e-6") for x in "42".split(",") ], time_threashold="2022-02-25 12:20:18", use_soft_score=use_soft_score)
# ----------- # clear_and_print(all_results)
# ----------- # all_results = []

# ----------- # get_log_path = lambda prefix, task_name: f"logs/bart0-zeroshot_Random/RET_Random-unsupervised_rerank(mean)-5-32-512-{prefix[0]}-{task_name}-{prefix[1]}-2.log"  
# ----------- # all_results +=  process_exp(get_log_path, [(x, "6e-6") for x in "42".split(",") ], time_threashold="2022-02-25 12:20:18", use_soft_score=use_soft_score)
# ----------- # clear_and_print(all_results)
# ----------- # all_results = []

# ----------- # get_log_path = lambda prefix, task_name: f"logs/bart0-zeroshot_Random/RET_Random-unsupervised_rerank-10-32-512-{prefix[0]}-{task_name}-{prefix[1]}-2.log"  
# ----------- # all_results +=  process_exp(get_log_path, [(x, "6e-6") for x in "42".split(",") ], time_threashold="2022-02-25 12:20:18", use_soft_score=use_soft_score)
# ----------- # clear_and_print(all_results)
# ----------- # all_results = []


##  Finalized results 

# get_log_path = lambda prefix, task_name: f"logs/bart0-zeroshot_Random/RET_Random-unsupervised_no-1-16-512-{prefix[0]}-{task_name}-{prefix[1]}-2.log"  
# all_results +=  process_exp(get_log_path, [(x, "6e-6") for x in "42".split(",") ], time_threashold="2022-03-05 12:20:18", use_soft_score=use_soft_score)
# clear_and_print(all_results)
# all_results = []

# get_log_path = lambda prefix, task_name: f"logs/bart0-zeroshot_SentenceTransformer/RET_SentenceTransformer-unsupervised_no-1-16-512-{prefix[0]}-{task_name}-{prefix[1]}-2.log"  
# all_results +=  process_exp(get_log_path, [(x, "6e-6") for x in "42".split(",") ], time_threashold="2022-03-05 12:00:00", use_soft_score=use_soft_score)
# clear_and_print(all_results)
# all_results = []


# get_log_path = lambda prefix, task_name: f"logs/bart0-zeroshot_BART/RET_BART-unsupervised_no-1-16-512-{prefix[0]}-{task_name}-{prefix[1]}-2.log"  
# all_results +=  process_exp(get_log_path, [(x, "6e-6") for x in "42".split(",") ], time_threashold="2022-03-05 12:00:18", use_soft_score=use_soft_score)
# clear_and_print(all_results)
# all_results = []


# get_log_path = lambda prefix, task_name: f"logs/bart0-zeroshot_Random/RET_Random-unsupervised_rerank-2-16-512-{prefix[0]}-{task_name}-{prefix[1]}-2.log"  
# all_results +=  process_exp(get_log_path, [(x, "6e-6") for x in "42".split(",") ], time_threashold="2022-03-05 12:00:18", use_soft_score=use_soft_score)
# clear_and_print(all_results)
# all_results = []

# get_log_path = lambda prefix, task_name: f"logs/bart0-zeroshot_SentenceTransformer/RET_SentenceTransformer-unsupervised_rerank-2-16-512-{prefix[0]}-{task_name}-{prefix[1]}-2.log"  
# all_results +=  process_exp(get_log_path, [(x, "6e-6") for x in "42".split(",") ], time_threashold="2022-03-05 12:00:18", use_soft_score=use_soft_score)
# clear_and_print(all_results)
# all_results = []


# get_log_path = lambda prefix, task_name: f"logs/bart0-zeroshot_BART/RET_BART-unsupervised_rerank-2-16-512-{prefix[0]}-{task_name}-{prefix[1]}-2.log"  
# all_results +=  process_exp(get_log_path, [(x, "6e-6") for x in "42".split(",") ], time_threashold="2022-03-05 23:00:18", use_soft_score=use_soft_score)
# clear_and_print(all_results)
# all_results = []

# get_log_path = lambda prefix, task_name: f"logs/bart0-zeroshot_BART/RET_BART-unsupervised_rerank-2-16-512-{prefix[0]}-{task_name}-{prefix[1]}-2.log"  
# all_results +=  process_exp(get_log_path, [(x, "6e-6") for x in "42".split(",") ], time_threashold="2022-04-01 23:00:18", use_soft_score=use_soft_score)
# clear_and_print(all_results)
# all_results = []


#------------------------#

# get_log_path = lambda prefix, task_name: f"logs/bart0-zeroshot_Random/RET_Random-unsupervised_rerank_roberta_base-2-16-512-{prefix[0]}-{task_name}-{prefix[1]}-2.log"  
# all_results +=  process_exp(get_log_path, [(x, "6e-6") for x in "42".split(",") ], time_threashold="2022-04-01 23:00:18", use_soft_score=use_soft_score)
# clear_and_print(all_results)
# all_results = []


# get_log_path = lambda prefix, task_name: f"logs/bart0-zeroshot_SentenceTransformer/RET_SentenceTransformer-unsupervised_rerank_roberta_base-2-16-512-{prefix[0]}-{task_name}-{prefix[1]}-2.log"  
# all_results +=  process_exp(get_log_path, [(x, "6e-6") for x in "42".split(",") ], time_threashold="2022-04-01 23:00:18", use_soft_score=use_soft_score)
# clear_and_print(all_results)
# all_results = []


# get_log_path = lambda prefix, task_name: f"logs/bart0-zeroshot_BART/RET_BART-unsupervised_rerank-2-16-512-{prefix[0]}-{task_name}-{prefix[1]}-2.log"  
# all_results +=  process_exp(get_log_path, [(x, "6e-6") for x in "42".split(",") ], time_threashold="2022-04-01 23:00:18", use_soft_score=use_soft_score)
# clear_and_print(all_results)
# all_results = []



# get_log_path = lambda prefix, task_name: f"logs/bart0-zeroshot_BART/RET_BART-unsupervised_rerank_roberta_baseI2-2-16-512-{prefix[0]}-{task_name}-{prefix[1]}-2.log"  
# all_results +=  process_exp(get_log_path, [(x, "6e-6") for x in "42".split(",") ], time_threashold="2022-04-01 23:00:18", use_soft_score=use_soft_score)
# clear_and_print(all_results)
# all_results = []


# ========================================== Abalation 

# get_log_path = lambda prefix, task_name: f"logs/bart0-zeroshot_BART/RET_BART-unsupervised_rerank_roberta_base-2-16-512-{prefix[0]}-{task_name}-{prefix[1]}-2.log"  
# all_results +=  process_exp(get_log_path, [(x, "6e-6") for x in "42".split(",") ], time_threashold="2022-04-01 23:00:18", use_soft_score=use_soft_score)
# clear_and_print(all_results)
all_results = []

# get_log_path = lambda prefix, task_name: f"logs/bart0-zeroshot_BART/RET_BART-unsupervised_rerank_roberta_base-2-32-512-{prefix[0]}-{task_name}-{prefix[1]}-2.log"  
# all_results +=  process_exp(get_log_path, [(x, "6e-6") for x in "42".split(",") ], time_threashold="2022-04-08 23:00:18", use_soft_score=use_soft_score)
# clear_and_print(all_results)
# all_results = []

# get_log_path = lambda prefix, task_name: f"logs/bart0-zeroshot_BART/RET_BART-unsupervised_rerank_roberta_base-2-64-512-{prefix[0]}-{task_name}-{prefix[1]}-2.log"  
# all_results +=  process_exp(get_log_path, [(x, "6e-6") for x in "42".split(",") ], time_threashold="2022-04-08 23:00:18", use_soft_score=use_soft_score)
# clear_and_print(all_results)
# all_results = []



# get_log_path = lambda prefix, task_name: f"logs/bart0-zeroshot_BART/RET_BART-unsupervised_rerank_roberta_base-2-16-256-{prefix[0]}-{task_name}-{prefix[1]}-2.log"  
# all_results +=  process_exp(get_log_path, [(x, "6e-6") for x in "42".split(",") ], time_threashold="2022-04-08 23:00:18", use_soft_score=use_soft_score)
# clear_and_print(all_results)
# all_results = []

# get_log_path = lambda prefix, task_name: f"logs/bart0-zeroshot_BART/RET_BART-unsupervised_rerank_roberta_base-2-16-1024-{prefix[0]}-{task_name}-{prefix[1]}-2.log"  
# all_results +=  process_exp(get_log_path, [(x, "6e-6") for x in "42".split(",") ], time_threashold="2022-04-08 23:00:18", use_soft_score=use_soft_score)
# clear_and_print(all_results)
# all_results = []




# get_log_path = lambda prefix, task_name: f"logs/bart0-zeroshot_BART/RET_BART-unsupervised_rerank_roberta_base-2-1-512-{prefix[0]}-{task_name}-{prefix[1]}-2.log"  
# all_results +=  process_exp(get_log_path, [(x, "6e-6") for x in "42".split(",") ], time_threashold="2022-04-01 23:00:18", use_soft_score=use_soft_score)
# clear_and_print(all_results)
# all_results = []

# get_log_path = lambda prefix, task_name: f"logs/bart0-zeroshot_BART/RET_BART-unsupervised_rerank_roberta_base-2-8-512-{prefix[0]}-{task_name}-{prefix[1]}-2.log"  
# all_results +=  process_exp(get_log_path, [(x, "6e-6") for x in "42".split(",") ], time_threashold="2022-04-01 23:00:18", use_soft_score=use_soft_score)
# clear_and_print(all_results)
# all_results = []

# get_log_path = lambda prefix, task_name: f"logs/bart0-zeroshot_BART/RET_BART-unsupervised_rerank_roberta_base-2-16-512-{prefix[0]}-{task_name}-{prefix[1]}-2.log"  
# all_results +=  process_exp(get_log_path, [(x, "6e-6") for x in "42".split(",") ], time_threashold="2022-04-01 23:00:18", use_soft_score=use_soft_score)
# clear_and_print(all_results)
# all_results = []


# get_log_path = lambda prefix, task_name: f"logs/bart0-zeroshot_BART/RET_BART-unsupervised_rerank_roberta_base-2-32-512-{prefix[0]}-{task_name}-{prefix[1]}-2.log"  
# all_results +=  process_exp(get_log_path, [(x, "6e-6") for x in "42".split(",") ], time_threashold="2022-04-01 23:00:18", use_soft_score=use_soft_score)
# clear_and_print(all_results)
# all_results = []



get_log_path = lambda prefix, task_name: f"logs/bart0-zeroshot_BART/RET_BART-unsupervised_rerank_roberta_base-1.5-16-512-{prefix[0]}-{task_name}-{prefix[1]}-2.log"  
all_results +=  process_exp(get_log_path, [(x, "6e-6") for x in "42".split(",") ], time_threashold="2022-04-01 23:00:18", use_soft_score=use_soft_score)
clear_and_print(all_results)
all_results = []

get_log_path = lambda prefix, task_name: f"logs/bart0-zeroshot_BART/RET_BART-unsupervised_rerank_roberta_base-2-16-512-{prefix[0]}-{task_name}-{prefix[1]}-2.log"  
all_results +=  process_exp(get_log_path, [(x, "6e-6") for x in "42".split(",") ], time_threashold="2022-04-15 00:00:18", use_soft_score=use_soft_score)
clear_and_print(all_results)
all_results = []

get_log_path = lambda prefix, task_name: f"logs/bart0-zeroshot_BART/RET_BART-unsupervised_rerank_roberta_base-3-16-512-{prefix[0]}-{task_name}-{prefix[1]}-2.log"  
all_results +=  process_exp(get_log_path, [(x, "6e-6") for x in "42".split(",") ], time_threashold="2022-04-01 23:00:18", use_soft_score=use_soft_score)
clear_and_print(all_results)
all_results = []

get_log_path = lambda prefix, task_name: f"logs/bart0-zeroshot_BART/RET_BART-unsupervised_rerank_roberta_base-4-16-512-{prefix[0]}-{task_name}-{prefix[1]}-2.log"  
all_results +=  process_exp(get_log_path, [(x, "6e-6") for x in "42".split(",") ], time_threashold="2022-04-01 23:00:18", use_soft_score=use_soft_score)
clear_and_print(all_results)
all_results = []



# not good
# get_log_path = lambda prefix, task_name: f"logs/bart0-zeroshot_BART/RET_BART-unsupervised_rerank_bart0-2-16-512-{prefix[0]}-{task_name}-{prefix[1]}-2.log"  
# all_results +=  process_exp(get_log_path, [(x, "6e-6") for x in "42".split(",") ], time_threashold="2022-04-01 23:00:18", use_soft_score=use_soft_score)
# clear_and_print(all_results)
# all_results = []


# get_log_path = lambda prefix, task_name: f"logs/bart0-zeroshot_BART/RET_BART-unsupervised_rerank_roberta_baseI1-2-16-512-{prefix[0]}-{task_name}-{prefix[1]}-2.log"  
# all_results +=  process_exp(get_log_path, [(x, "6e-6") for x in "42".split(",") ], time_threashold="2022-04-01 23:00:18", use_soft_score=use_soft_score)
# clear_and_print(all_results)
# all_results = []

# get_log_path = lambda prefix, task_name: f"logs/bart0-zeroshot_BART/RET_BART-unsupervised_rerank_roberta_baseI01m-2-16-512-{prefix[0]}-{task_name}-{prefix[1]}-2.log"  
# all_results +=  process_exp(get_log_path, [(x, "6e-6") for x in "42".split(",") ], time_threashold="2022-04-01 23:00:18", use_soft_score=use_soft_score)
# clear_and_print(all_results)
# all_results = []


# big bench 

# print("BART0+Rerank")
# get_log_path = lambda prefix, task_name: f"logs/bart0-zeroshot_BART/RET_BART-unsupervised_rerank_roberta_base-2-16-512-{prefix[0]}-{task_name}-{prefix[1]}-2.log"  
# all_results +=  process_exp(get_log_path, [(x, "6e-6") for x in "42".split(",") ], time_threashold="2022-04-12 23:00:18", use_soft_score=use_soft_score)
# clear_and_print(all_results)
# all_results = []

# print("BART0")
# get_log_path = lambda prefix, task_name: f"logs/bart0-zeroshot_BART/RET_BART-unsupervised_no_none-1-16-512-{prefix[0]}-{task_name}-{prefix[1]}-2.log"  
# all_results +=  process_exp(get_log_path, [(x, "6e-6") for x in "42".split(",") ], time_threashold="2022-04-12 23:00:18", use_soft_score=use_soft_score)
# clear_and_print(all_results)
# all_results = []

# print("SBERT")
# get_log_path = lambda prefix, task_name: f"logs/bart0-zeroshot_SentenceTransformer/RET_SentenceTransformer-unsupervised_no_none-1-16-512-{prefix[0]}-{task_name}-{prefix[1]}-2.log"  
# all_results +=  process_exp(get_log_path, [(x, "6e-6") for x in "42".split(",") ], time_threashold="2022-04-12 23:00:18", use_soft_score=use_soft_score)
# clear_and_print(all_results)
# all_results = []

# print("Random")
# get_log_path = lambda prefix, task_name: f"logs/bart0-zeroshot_Random/RET_Random-unsupervised_no_none-1-16-512-{prefix[0]}-{task_name}-{prefix[1]}-2.log"  
# all_results +=  process_exp(get_log_path, [(x, "6e-6") for x in "42".split(",") ], time_threashold="2022-04-12 23:00:18", use_soft_score=use_soft_score)
# clear_and_print(all_results)
# all_results = []


 


# removal abalation 
# remove = "none"
# groupnames = [
#     "none",
#     "multiple_choice_qa",
#     "summarization",
#     "extractive_qa",
#     "sentiment",
#     "closed_book_qa",
#     "structure_to_text",
#     "topic_classification",
#     "paraphrase_identification",
# ]
# for remove in groupnames:
#     print(f"----{remove}-----")
#     get_log_path = lambda prefix, task_name: f"logs/bart0-zeroshot_BART/RET_BART-unsupervised_rerank_roberta_base-{remove}-2-16-512-{prefix[0]}-{task_name}-{prefix[1]}-2.log"  
#     all_results +=  process_exp(get_log_path, [(x, "6e-6") for x in "42".split(",") ], time_threashold="2022-04-01 23:00:18", use_soft_score=use_soft_score)
#     clear_and_print(all_results)
#     all_results = []


# =======================================

# get_log_path = lambda prefix, task_name: f"logs/bart0-zeroshot_BART/RET_BART-unsupervised_rerank_roberta_base-3-16-512-{prefix[0]}-{task_name}-{prefix[1]}-2.log"  
# all_results +=  process_exp(get_log_path, [(x, "6e-6") for x in "42".split(",") ], time_threashold="2022-04-01 23:00:18", use_soft_score=use_soft_score)
# clear_and_print(all_results)
# all_results = []

# get_log_path = lambda prefix, task_name: f"logs/bart0-zeroshot_BART/RET_BART-unsupervised_rerank_bart0_base-2-16-512-{prefix[0]}-{task_name}-{prefix[1]}-2.log"  
# all_results +=  process_exp(get_log_path, [(x, "6e-6") for x in "42".split(",") ], time_threashold="2022-04-01 23:00:18", use_soft_score=use_soft_score)
# clear_and_print(all_results)
# all_results = []

exit()

# get_log_path = lambda prefix, task_name: f"logs/bart0-zeroshot_BART/RET_BART-unsupervised_rerank-2-16-512-{prefix[0]}-{task_name}-{prefix[1]}-2.log"  
# all_results +=  process_exp(get_log_path, [(x, "6e-6") for x in "42".split(",") ], time_threashold="2022-03-06 00:00:18", use_soft_score=use_soft_score)
# clear_and_print(all_results)
# all_results = []


# get_log_path = lambda prefix, task_name: f"logs/bart0-zeroshot_SentenceTransformer/RET_SentenceTransformer-unsupervised_rerank-3-16-512-{prefix[0]}-{task_name}-{prefix[1]}-2.log"  
# all_results +=  process_exp(get_log_path, [(x, "6e-6") for x in "42".split(",") ], time_threashold="2022-03-02 12:00:18", use_soft_score=use_soft_score)
# clear_and_print(all_results)
# all_results = []


# get_log_path = lambda prefix, task_name: f"logs/bart0-zeroshot_TwoStage-BART-TRAINED/RET_TwoStage-BART-TRAINED-unsupervised_no-2-16-512-{prefix[0]}-{task_name}-{prefix[1]}-2.log"  
# all_results +=  process_exp(get_log_path, [(x, "6e-6") for x in "42".split(",") ], time_threashold="2022-03-02 12:00:18", use_soft_score=use_soft_score)
# clear_and_print(all_results)
# all_results = []

exit()


# -----------------------------------
print("-"*100)
get_log_path = lambda prefix, task_name: f"logs/bart0-zeroshot_Random/RET_Random-unsupervised_no-1-16-512-{prefix[0]}-{task_name}-{prefix[1]}-2.log"  
all_results +=  process_exp(get_log_path, [(x, "6e-6") for x in "2022".split(",") ], time_threashold="2022-03-02 12:00:18", use_soft_score=use_soft_score)
clear_and_print(all_results)
all_results = []
 
get_log_path = lambda prefix, task_name: f"logs/bart0-zeroshot_SentenceTransformer/RET_SentenceTransformer-unsupervised_no-1-16-512-{prefix[0]}-{task_name}-{prefix[1]}-2.log"  
all_results +=  process_exp(get_log_path, [(x, "6e-6") for x in "2022".split(",") ], time_threashold="2022-03-02 12:00:18", use_soft_score=use_soft_score)
clear_and_print(all_results)
all_results = []


get_log_path = lambda prefix, task_name: f"logs/bart0-zeroshot_BART/RET_BART-unsupervised_no-1-16-512-{prefix[0]}-{task_name}-{prefix[1]}-2.log"  
all_results +=  process_exp(get_log_path, [(x, "6e-6") for x in "2022".split(",") ], time_threashold="2022-03-03 12:00:18", use_soft_score=use_soft_score)
clear_and_print(all_results)
all_results = []
 

# print("*"*50)

# # get_log_path = lambda prefix, task_name: f"logs/bart0-zeroshot_Random/RET_Random-unsupervised_rerank-3-16-512-{prefix[0]}-{task_name}-{prefix[1]}-2.log"  
# # all_results +=  process_exp(get_log_path, [(x, "6e-6") for x in "2022".split(",") ], time_threashold="2022-03-02 12:00:18", use_soft_score=use_soft_score)
# # clear_and_print(all_results)
# # all_results = []
 


# # get_log_path = lambda prefix, task_name: f"logs/bart0-zeroshot_SentenceTransformer/RET_SentenceTransformer-unsupervised_rerank-3-16-512-{prefix[0]}-{task_name}-{prefix[1]}-2.log"  
# # all_results +=  process_exp(get_log_path, [(x, "6e-6") for x in "2022".split(",") ], time_threashold="2022-03-02 12:00:18", use_soft_score=use_soft_score)
# # clear_and_print(all_results)
# # all_results = []

# # get_log_path = lambda prefix, task_name: f"logs/bart0-zeroshot_BART/RET_BART-unsupervised_rerank-3-16-512-{prefix[0]}-{task_name}-{prefix[1]}-2.log"  
# # all_results +=  process_exp(get_log_path, [(x, "6e-6") for x in "2022".split(",") ], time_threashold="2022-03-02 12:00:18", use_soft_score=use_soft_score)
# # clear_and_print(all_results)
# # all_results = []




# get_log_path = lambda prefix, task_name: f"logs/bart0-zeroshot_vBART/RET_vBART-unsupervised_no-1-16-512-{prefix[0]}-{task_name}-{prefix[1]}-2.log"  
# all_results +=  process_exp(get_log_path, [(x, "6e-6") for x in "2022".split(",") ], time_threashold="2022-03-02 12:00:18", use_soft_score=use_soft_score)
# clear_and_print(all_results)
# all_results = []



# -------------------------------------





# get_log_path = lambda prefix, task_name: f"logs/bart0-zeroshot_BART/RET_BART-unsupervised_no-1-16-512-{prefix[0]}_scores-{task_name}-{prefix[1]}-2.log"  
# all_results +=  process_exp(get_log_path, [(x, "6e-6") for x in "42".split(",") ], time_threashold="2022-03-01 12:00:18", use_soft_score=use_soft_score)
# clear_and_print(all_results)
# all_results = []



# get_log_path = lambda prefix, task_name: f"logs/bart0-zeroshot_BART/RET_BART-unsupervised_no-1-32-512-{prefix[0]}-{task_name}-{prefix[1]}-2.log"  
# all_results +=  process_exp(get_log_path, [(x, "6e-6") for x in "42".split(",") ], time_threashold="2022-03-01 12:00:18", use_soft_score=use_soft_score)
# clear_and_print(all_results)
# all_results = []

# get_log_path = lambda prefix, task_name: f"logs/bart0-zeroshot_SentenceTransformer/RET_SentenceTransformer-unsupervised_rerank-5-32-512-{prefix[0]}-{task_name}-{prefix[1]}-2.log"  
# all_results +=  process_exp(get_log_path, [(x, "6e-6") for x in "42".split(",") ], time_threashold="2022-03-01 12:00:00", use_soft_score=use_soft_score)
# clear_and_print(all_results)
# all_results = []





# get_log_path = lambda prefix, task_name: f"logs/bart0-zeroshot_BART/RET_BART-unsupervised_rerank-3-32-512-{prefix[0]}-{task_name}-{prefix[1]}-2.log"  
# all_results +=  process_exp(get_log_path, [(x, "6e-6") for x in "42".split(",") ], time_threashold="2022-03-01 12:00:18", use_soft_score=use_soft_score)
# clear_and_print(all_results)
# all_results = []

# -------------------------------
# get_log_path = lambda prefix, task_name: f"logs/bart0-zeroshot_TRAINED/RET_TRAINED-unsupervised_rerank-5-32-512-{prefix[0]}-{task_name}-{prefix[1]}-2.log"  
# all_results +=  process_exp(get_log_path, [(x, "6e-6") for x in "42".split(",") ], time_threashold="2022-02-25 12:20:18", use_soft_score=use_soft_score)
# clear_and_print(all_results)
# all_results = []

# get_log_path = lambda prefix, task_name: f"logs/bart0-zeroshot_SentenceTransformer/RET_SentenceTransformer-unsupervised_rerank-5-32-512-{prefix[0]}-{task_name}-{prefix[1]}-2.log"  
# all_results +=  process_exp(get_log_path, [(x, "6e-6") for x in "42".split(",") ], time_threashold="2022-02-25 12:20:18", use_soft_score=use_soft_score)
# clear_and_print(all_results)
# all_results = []


# get_log_path = lambda prefix, task_name: f"logs/bart0-zeroshot_BART/RET_BART-unsupervised_rerank-5-32-512-{prefix[0]}-{task_name}-{prefix[1]}-2.log"  
# all_results +=  process_exp(get_log_path, [(x, "6e-6") for x in "42".split(",") ], time_threashold="2022-02-25 12:20:18", use_soft_score=use_soft_score)
# clear_and_print(all_results)
# all_results = []



# get_log_path = lambda prefix, task_name: f"logs/bart0-zeroshot_BART/RET_BART-unsupervised_no-1-32-1024-{prefix[0]}-{task_name}-{prefix[1]}-2.log"  
# all_results +=  process_exp(get_log_path, [(x, "6e-6") for x in "42".split(",") ], time_threashold="2022-02-25 12:20:18", use_soft_score=use_soft_score)
# clear_and_print(all_results)
# all_results = []


# get_log_path = lambda prefix, task_name: f"logs/bart0-zeroshot_BART/RET_BART-unsupervised_no-1-64-512-{prefix[0]}-{task_name}-{prefix[1]}-2.log"  
# all_results +=  process_exp(get_log_path, [(x, "6e-6") for x in "42".split(",") ], time_threashold="2022-02-28 12:20:18", use_soft_score=use_soft_score)
# clear_and_print(all_results)
# all_results = []




# get_log_path = lambda prefix, task_name: f"logs/bart0-zeroshot_BART/RET_BART-unsupervised_rerank-10-32-512-{prefix[0]}-{task_name}-{prefix[1]}-2.log"  
# all_results +=  process_exp(get_log_path, [(x, "6e-6") for x in "42".split(",") ], time_threashold="2022-02-25 12:20:18", use_soft_score=use_soft_score)
# clear_and_print(all_results)
# all_results = []


# get_log_path = lambda prefix, task_name: f"logs/bart0-zeroshot_TRAINED/RET_TRAINED-unsupervised_no-1-32-512-{prefix[0]}-{task_name}-{prefix[1]}-2.log"  
# all_results +=  process_exp(get_log_path, [(x, "6e-6") for x in "42".split(",") ], time_threashold="2022-02-25 12:20:18", use_soft_score=use_soft_score)
# clear_and_print(all_results)
# all_results = []


# get_log_path = lambda prefix, task_name: f"logs/bart0-zeroshot_Random/RET_Random-unsupervised_rerank(max)-5-32-512-{prefix[0]}-{task_name}-{prefix[1]}-2.log"  
# all_results +=  process_exp(get_log_path, [(x, "6e-6") for x in "42".split(",") ], time_threashold="2022-02-25 12:20:18", use_soft_score=use_soft_score)
# clear_and_print(all_results)
# all_results = []



# get_log_path = lambda prefix, task_name: f"logs/bart0-zeroshot_Random/RET_Random-unsupervised_rerank(sm)-3-32-512-{prefix[0]}-{task_name}-{prefix[1]}-2.log"  
# all_results +=  process_exp(get_log_path, [(x, "6e-6") for x in "42".split(",") ], time_threashold="2022-02-25 12:20:18", use_soft_score=use_soft_score)
# clear_and_print(all_results)
# all_results = []

# get_log_path = lambda prefix, task_name: f"logs/bart0-zeroshot_Random/RET_Random-unsupervised_rerank-10-32-512-{prefix[0]}-{task_name}-{prefix[1]}-2.log"  
# all_results +=  process_exp(get_log_path, [(x, "6e-6") for x in "42".split(",") ], time_threashold="2022-02-22 20:20:18", use_soft_score=use_soft_score)
# clear_and_print(all_results)
# all_results = []



# get_log_path = lambda prefix, task_name: f"logs/bart0-zeroshot_SentenceTransformer/RET_SentenceTransformer-unsupervised_rerank-5-32-512-{prefix[0]}-{task_name}-{prefix[1]}-2.log"  
# all_results +=  process_exp(get_log_path, [(x, "6e-6") for x in "42".split(",") ], time_threashold="2022-02-21 10:20:18", use_soft_score=use_soft_score)
# clear_and_print(all_results)
# all_results = []


# get_log_path = lambda prefix, task_name: f"logs/bart0-zeroshot_Random/RET_Random-unsupervised_rerank-64-512-{prefix[0]}-all-{prefix[1]}-1.log"  
# all_results +=  process_exp(get_log_path, [(x, "5e-6") for x in "42,1337,2333,2022,1213,43,1338,2334,2023,1214".split(",") ], time_threashold="2022-02-21 00:27:18", use_soft_score=use_soft_score)
# clear_and_print(all_results)
# all_results = []

# get_log_path = lambda prefix, task_name: f"logs/bart0-zeroshot_SentenceTransformer/RET_SentenceTransformer-unsupervised_rerank-64-1024-{prefix[0]}-all-{prefix[1]}-1.log"  
# all_results +=  process_exp(get_log_path, [(x, "5e-6") for x in "42,1337,2333,2022,1213,43,1338,2334,2023,1214".split(",") ], time_threashold="2022-02-12 00:27:18", use_soft_score=use_soft_score)
# clear_and_print(all_results)
# all_results = []

# clear_and_print(all_results)
# all_results = []

# # For Few-Shot Evalution
# get_log_path = lambda prefix, task_name: f"logs_backup/bart0-fewshot/FS_none-{prefix[0]}-{prefix[2]}-{task_name}-{prefix[1]}-{prefix[3]}.log"   # for Few-shot
# all_results +=  process_exp(get_log_path, [(64, "1e-5", "True", 10)], time_threashold="2022-01-24 12:27:18", use_soft_score=use_soft_score)
# clear_and_print(all_results)
# all_results = []

# get_log_path = lambda prefix, task_name: f"logs/bart0-fewshot_Random/RET_Random-two-stage_no-{prefix[0]}-1024-42-{task_name}-5e-6-2.log"   # for Few-shot
# all_results +=  process_exp(get_log_path, [("64", )], time_threashold="2022-02-11 00:27:18", use_soft_score=use_soft_score)


# clear_and_print(all_results)
# all_results = []

# get_log_path = lambda prefix, task_name: f"logs/bart0-fewshot_Random/RET_Random-two-stage_rerank-{prefix[0]}-1024-42-{task_name}-5e-6-2.log"   # for Few-shot
# all_results +=  process_exp(get_log_path, [("64", )], time_threashold="2022-02-15 00:27:18", use_soft_score=use_soft_score)


# clear_and_print(all_results)

# get_log_path = lambda prefix, task_name: f"logs/bart0-fewshot_Random/RET_Random-mix-{prefix[0]}-512-42-{task_name}-6e-6-2.log"   # for Few-shot
# all_results +=  process_exp(get_log_path, [("64", )], time_threashold="2022-01-11 00:27:18")

# logs/bart0-fewshot_Random/RET_Random-two-stage-64-512-42-ai2_arc-ARC-Challenge-6e-6-2.log




# # # For Zero-Shot + Random Ret Evalution
# get_log_path = lambda prefix, task_name: f"logs/BART0-zeroshot-retriever/random-{prefix[0]}-{task_name}/BART0-Random-{prefix[0]}-3331-3e-6-{prefix[1]}-zs-{task_name}.log"   # for Few-shot
# all_results +=  process_exp(get_log_path, [(42, 200), (1337, 200), (2022, 200), (1213, 200), (2333, 200),])

# --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# For CSR-related

# all_results += process_exp(get_log_path, ["BART0_CSR"], time_threashold="2022-01-25 00:27:18", use_soft_score=use_soft_score) # ,  "T0_3B"

# get_log_path = lambda prefix, task_name: f"logs/csr-fewshot/FS_BART_none-{prefix[0]}-{prefix[2]}-{task_name}-{prefix[1]}-{prefix[3]}.log"   # for Few-shot
# all_results +=  process_exp(get_log_path, [
#                                             # (64, "1e-5", "True", 10, "v1"),
#                                             (64, "1e-5", "True", 5, "v1")], time_threashold="2022-01-24 12:27:18")

# get_log_path = lambda prefix, task_name: f"logs/csr-fewshot/FS_BART0_CSR_none-{prefix[0]}-{prefix[2]}-{task_name}-{prefix[1]}-{prefix[3]}.log"   # for Few-shot
# all_results +=  process_exp(get_log_path, [(64, "1e-5", "True", 5, "v2")], time_threashold="2022-01-24 12:27:18")

# get_log_path = lambda prefix, task_name: f"logs/csr-fewshot_Random/RET_BART_Random-two-stage-{prefix[0]}-512-42-{task_name}-6e-6-2.log"   # for Few-shot
# all_results +=  process_exp(get_log_path, [("64", "Random" )], time_threashold="2022-01-11 00:27:18")

# get_log_path = lambda prefix, task_name: f"logs/csr-fewshot_{prefix[1]}/RET_BART0_CSR_{prefix[1]}-two-stage-{prefix[0]}-512-42-{task_name}-1e-5-3.log"   # for Few-shot
# all_results +=  process_exp(get_log_path, [("64", "Random")], time_threashold="2022-01-11 00:27:18")
# all_results +=  process_exp(get_log_path, [("64", "SentenceTransformer")], time_threashold="2022-01-11 00:27:18")
# all_results +=  process_exp(get_log_path, [("64", "BART")], time_threashold="2022-01-11 00:27:18")






# Viz
# all_results_pd = pd.DataFrame(all_results).sort_values(by=['task']).drop_duplicates().groupby("task").sum().reset_index()




# all_results_pd = pd.DataFrame(all_results).sort_values(by=['task']).drop_duplicates().groupby("task").sum().reset_index()

# pd.options.display.float_format = '{:.2%}'.format

# print(all_results_pd)
# print(all_results_pd.to_csv(index=False))


# print(all_results_pd[["task", "BART0", "Random-0s@0", "SBERT-0s@0", "BART-0s@0"]].to_csv(index=False))
# print(all_results_pd[["task", "BART0", "Random-0s@1", "SBERT-0s@1", "BART-0s@1"]].to_csv(index=False))
# print(all_results_pd[["task", "BART0", "Random-0s@2", "SBERT-0s@2", "BART-0s@2"]].to_csv(index=False))
# print(all_results_pd[["task", "BART0", "Random-0s@3", "SBERT-0s@3", "BART-0s@3"]].to_csv(index=False))
# print(all_results_pd[["task", "BART0", "Random-0s@4", "SBERT-0s@4", "BART-0s@4"]].to_csv(index=False))
