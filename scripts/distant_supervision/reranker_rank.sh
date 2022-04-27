#!/bin/bash
#SBATCH --time=3:00:00
#SBATCH --job-name=infer_eval
#SBATCH --output jobs/%j.out
#SBATCH --gres=gpu:1 

python metax/distant_supervision/train_reranker.py \
--runname rank_on_fs \
--mode infer \
--reranker_model_path checkpoints/reranker/roberta_reranker_v2/checkpoint-4500 \
--input_data_path data/ds_copy/roberta_reranker_v2/eval.json \
--output_data_path data/ds_copy/roberta_reranker_v2/