#!/bin/bash
#SBATCH --time=5:00:00
#SBATCH --job-name=compare
#SBATCH --output jobs/%j.log
#SBATCH --gres=gpu:6000:1

RELA_PATH=$(scontrol show job $SLURM_JOBID | awk -F= '/Command=/{print $2}')
SCRIPT_PATH=$(realpath $(dirname "${RELA_PATH}"))

source $SCRIPT_PATH/reranker_config.sh

python metax/reranker_bootstrap/compare_rerankers.py \
    --checkpoint_A checkpoints/reranker/iteration0_v1 \
    --checkpoint_B checkpoints/reranker/iteration1_v1 \
    --retrieved_data_path data/bootstrap_reranker/bart_retrieved/query_retrieved_1_outof_6.json \