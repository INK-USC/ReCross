#!/bin/bash
#SBATCH --time=5:00:00
#SBATCH --job-name=train_reranker
#SBATCH --output jobs/%j.out
#SBATCH --gres=gpu:2080:8
#SBATCH --exclude ink-molly

python metax/distant_supervision/train_reranker.py \
    --ds_train_data_path "data/ds_from_bart0_upstream_train.json" \
    --ds_dev_data_path "data/ds_from_bart0_upstream_test.json" \
    --ds_test_data_path "data/ds_from_bart0_upstream_dev.json" \
    --ds_train_cut_off -1 \
    --ds_eval_cut_off -1 \
    --partitioned_data_path "data/ds_copy/" \
    --reranker_model_path "checkpoints/reranker/" \
    --runname "roberta_reranker_v8" \
    --mode train
    

# sbatch scripts/distant_supervision/train_reranker.sh