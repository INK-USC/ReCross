#!/bin/bash
#SBATCH --time=5:00:00
#SBATCH --job-name=tune
#SBATCH --output jobs/%j.log
#SBATCH --gres=gpu:6000:3

RELA_PATH=$(scontrol show job $SLURM_JOBID | awk -F= '/Command=/{print $2}')
SCRIPT_PATH=$(realpath $(dirname "${RELA_PATH}"))

source $SCRIPT_PATH/reranker_config.sh

python metax/reranker_bootstrap/tune_reranker.py \
    --train_tuple_path generated_data/iteration0_v1/train_tuples.json \
    --dev_tuple_path generated_data/iteration0_v1/dev_tuples.json \
    --checkpoint_save_path checkpoints/reranker/$RERANKER_MODEL_ALIAS/iteration0/ \
    --reranker_model_type $RERANKER_MODEL_TYPE \
    --reranker_model_name $RERANKER_MODEL_NAME 