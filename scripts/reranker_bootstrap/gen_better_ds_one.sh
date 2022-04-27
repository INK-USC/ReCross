#!/bin/bash
#SBATCH --time=6:00:00
#SBATCH --job-name=one_gen
#SBATCH --output jobs/%j.log
#SBATCH --gres=gpu:6000:1

SHARD_ID=$1
SHARD_N=$2

RELA_PATH=$(scontrol show job $SLURM_JOBID | awk -F= '/Command=/{print $2}')
SCRIPT_PATH=$(realpath $(dirname "${RELA_PATH}"))

source $SCRIPT_PATH/reranker_config.sh

python metax/reranker_bootstrap/gen_better_ds.py \
    --initial_checkpoint_path checkpoints/reranker/iteration0_v1 \
    --query_and_candidate_data_path data/bootstrap_reranker/bart_retrieved/query_retrieved_$1_outof_$2.json \
    --data_save_path generated_data/iteration1_v1/better_ds_$1_outof_$2.json/ \
    --upstream_model_type yuchenlin/BART0 \
    --reranker_model_type $RERANKER_MODEL_TYPE \ 
    --reranker_model_name $RERANKER_MODEL_NAME \ 