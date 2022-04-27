#!/bin/bash
#SBATCH --time=10:00:00
#SBATCH --job-name=metax
#SBATCH --output jobs/%j.out
#SBATCH --gres=gpu:6000:2
#SBATCH --exclude ink-molly
RETRIEVE_NUM=1000
EARLY_STOPPING=true
TRAIN_SEED=3331
TARGET_TASK=$1
LR=1e-6

LOG_DIR="logs/bart0pp-zeroshot-bart-retriever"
RESULT_DIR="results/bart0pp-zeroshot-bart-retriever"
CKPT_DIR="checkpoints"
mkdir -p $LOG_DIR
mkdir -p $RESULT_DIR
mkdir -p $CKPT_DIR

export TOKENIZERS_PARALLELISM=false

model_seeds=($(cat scripts/zs_retrieval/retrieve.seeds))
for seed in "${model_seeds[@]}"
do
  echo "Running for dataset ${TARGET_TASK} with seed ${seed} "
  python -m metax.run \
      --run_name="${SHOTS}-${RETRIEVE_NUM}-${EARLY_STOPPING}-${TARGET_TASK}-${LR}" \
      --use_retriever --action="zs_retrieve_pipeline" \
      --upstream_train_file "data/t0pp_upstream_train_lines.json" \
      --memory_cache_path "memory_cache/bart0pp_memory.pkl" \
      --log_dir=$LOG_DIR \
      --checkpoint=$CKPT_DIR \
      --num_train_epochs=1 \
      --learning_rate=$LR \
      --train_batch_size=8 \
      --predict_batch_size=8 \
      --target_task=$TARGET_TASK \
      --output_dir=$RESULT_DIR \
      --early_stopping=$EARLY_STOPPING \
      --max_input_length=128 \
      --model_type='yuchenlin/BART0pp' \
      --upstream_num_shots=$RETRIEVE_NUM \
      --use_retriever \
      --train_seed $TRAIN_SEED \
      --retrieve_seed $seed \
      --retriever_mode "bart" \
      --query_aggregation_mode "aggregate_choices"
done
