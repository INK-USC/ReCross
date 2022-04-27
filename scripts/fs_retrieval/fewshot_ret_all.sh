#!/bin/bash
# data_names=($(cat scripts/downstream.tasks))
data_names=($(cat scripts/all_downstream.tasks))
ret="SentenceTransformer"
mode="mix"
for data_name in "${data_names[@]}" 
do
    sbatch scripts/fs_retrieval/fewshot_ret_one.sh ${data_name} ${ret} ${mode}
    echo "Submitted: ${data_name}"
done 