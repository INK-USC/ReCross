#!/bin/bash
model_seeds=($(cat scripts/zs_retrieval/retrieve.seeds))
for seed in "${model_seeds[@]}" 
do
    sbatch scripts/zs_retrieval/zeroshot_random_tune_one.sh ${seed}
    echo "Submitted Retrieval Fine-tune of Model ${seed}" 
done    