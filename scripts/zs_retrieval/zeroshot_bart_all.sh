#!/bin/bash
data_names=($(cat scripts/downstream.tasks))

for data_name in "${data_names[@]}"
do
    sbatch scripts/zs_retrieval/zeroshot_bart_one.sh ${data_name}
    echo "Submitted run for ${data_name} "
done
