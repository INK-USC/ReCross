#!/bin/bash
# data_names=($(cat scripts/downstream.tasks))
data_names=($(cat scripts/all_downstream.tasks))
# model_seeds=($(cat scripts/zs_retrieval/retrieve.seeds)) # too many
# declare -a model_seeds=("42")
declare -a model_seeds=("42" "2022" "1213" "1337" "2333")
for seed in "${model_seeds[@]}"
do
    for data_name in "${data_names[@]}"
    do
        sbatch scripts/zs_retrieval/zeroshot_semantic_one.sh ${data_name} ${seed}
        echo "Submitted Model ${seed} on ${data_name} "
    done
done
