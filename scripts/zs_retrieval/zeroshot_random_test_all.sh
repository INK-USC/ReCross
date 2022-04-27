#!/bin/bash 
data_names=($(cat scripts/all_downstream.tasks))
# model_seeds=($(cat scripts/zs_retrieval/retrieve.seeds))
declare -a model_seeds=("1213" "1337")
for seed in "${model_seeds[@]}" 
do
    for data_name in "${data_names[@]}" 
    do
        sbatch scripts/zs_retrieval/zeroshot_random_test_one.sh ${data_name} ${seed}
        echo "Submitted Evaluating model ${seed} on ${data_name} "
    done 
done    