#!/bin/bash
#SBATCH --time=6:00:00
#SBATCH --job-name=metax_fs_eval
#SBATCH --output jobs/%j.out
#SBATCH --gres=gpu:6000:1
#SBATCH --exclude ink-molly



EARLY_STOPPING=False
TARGET_TASK=$1
LR=1e-5
RETLR=3e-6
EPOCH=3
MODEL="yuchenlin/BART0"
RETRIEVER=$2 # "Random"
MODE=$3 # "two-stage"
SHOTS=64
RETRIEVE_NUM=512
LOG_DIR="logs/bart0-fewshot_${RETRIEVER}"
RESULT_DIR="results/bart0-fewshot_${RETRIEVER}"
CKPT_DIR="checkpoints"
mkdir -p $LOG_DIR
mkdir -p $RESULT_DIR
mkdir -p $CKPT_DIR


if [[ "$RETRIEVER" == "SentenceTransformer"* ]]; then
    memory_cache_path="memory_cache/sbert_memory.pkl"
elif [[ "$RETRIEVER" == "Random"* ]]; then
    memory_cache_path="memory_cache/random_memory.pkl"
elif [[ "$RETRIEVER" == "BART"* ]]; then
  if [[ "$MODE" == "unsupervised" ]]; then
    memory_cache_path="memory_cache/bart0_memory_zeroshot.pkl"
  else
    memory_cache_path="memory_cache/bart0_memory_fewshot.pkl"
  fi
elif [[ "$RETRIEVER" == "Trained" ]]; then
  if [[ "$MODE" == "unsupervised" ]]; then
    memory_cache_path="memory_cache/trained_memory_zeroshot.pkl"
  else
    memory_cache_path="memory_cache/trained_memory_zeroshot.pkl"
  fi
fi


python -m metax.run \
    --run_name "FS_${RETRIEVER}-${MODE}-${SHOTS}-${RETRIEVE_NUM}-${EARLY_STOPPING}-${TARGET_TASK}-${LR}-${EPOCH}" \
    --use_retriever --retriever_mode $RETRIEVER \
    --ret_merge_mode $MODE \
    --memory_cache_path $memory_cache_path \
    --action "few_shot_evaluation" \
    --query_aggregation_mode "aggregate_choices" \
    --log_dir $LOG_DIR \
    --checkpoint $CKPT_DIR \
    --num_train_epochs ${EPOCH} \
    --num_shots $SHOTS \
    --warmup_steps 0 \
    --learning_rate $LR \
    --ret_learning_rate $RETLR \
    --train_batch_size 4 \
    --predict_batch_size 8 \
    --target_task $TARGET_TASK \
    --output_dir $RESULT_DIR \
    --early_stopping $EARLY_STOPPING \
    --max_input_length 444 \
    --model_type ${MODEL} \
    --test_on_the_fly True \
    --upstream_num_shots ${RETRIEVE_NUM} \
    --finetune_round 3

    # \
    # --finetune_layers=1

# sbatch scripts/fs_retrieval/fewshot_ret_one.sh super_glue-cb Random mix
# sbatch scripts/fs_retrieval/fewshot_ret_one.sh super_glue-cb Random two-stage
# sbatch scripts/fs_retrieval/fewshot_ret_one.sh super_glue-cb SentenceTransformer two-stage
# sbatch scripts/fs_retrieval/fewshot_ret_one.sh super_glue-cb SentenceTransformer mix
# sbatch scripts/fs_retrieval/fewshot_ret_one.sh super_glue-cb BART two-stage
# sbatch scripts/fs_retrieval/fewshot_ret_one.sh super_glue-cb BART mix
