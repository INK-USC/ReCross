#!/bin/bash
#SBATCH --time=1:00:00
#SBATCH --job-name=metax_zs_train_random
#SBATCH --output jobs/%j.out
#SBATCH --gres=gpu:6000:1
#SBATCH --exclude ink-molly

RETRIEVE_NUM=500
EARLY_STOPPING=False
TARGET_TASK=super_glue-cb # A placeholder. Not actually used.  Only when test_on_the_fly=True
RETRIEVE_SEED=$1
TRAIN_SEED=3331
LR=3e-6    # try
EPOCH=10
TOTAL_STEPS=200
RETRIEVER="Random"
MODEL_NAME="BART0"
MODEL_URL="yuchenlin/BART0"
SAVE_CHECKPOINT=true


LOG_DIR="logs/zeroshot_${RETRIEVER}"
RETRIEVED_DATA_DIR="retrieved_data/zeroshot_${RETRIEVER}" # dir to save data retrieved by retrievers
RESULT_DIR="results/zeroshot_${RETRIEVER}"
CKPT_DIR="checkpoints"
CKPT_PREFIX="${MODEL_NAME}-${RETRIEVER}-${RETRIEVE_SEED}-${TRAIN_SEED}-${LR}"

mkdir -p $LOG_DIR
mkdir -p $RESULT_DIR
mkdir -p $CKPT_DIR
mkdir -p $RETRIEVED_DATA_DIR

export TOKENIZERS_PARALLELISM=false

python -m metax.run \
    --run_name "${RETRIEVER}-${RETRIEVE_NUM}-${RETRIEVE_SEED}-${EARLY_STOPPING}-${TARGET_TASK}-${LR}-${TOTAL_STEPS}" \
    --use_retriever \
    --action "zs_retrieve_pipeline" \
    --upstream_train_file "data/bart0_upstream_train_lines.json" \
    --memory_cache_path "memory_cache/${MODEL_NAME}_random_memory.pkl" \
    --log_dir $LOG_DIR \
    --checkpoint $CKPT_DIR \
    --checkpoint_prefix $CKPT_PREFIX \
    --total_steps $TOTAL_STEPS \
    --num_train_epochs $EPOCH \
    --ret_learning_rate $LR \
    --train_batch_size 4 \
    --predict_batch_size 8 \
    --target_task $TARGET_TASK \
    --output_dir $RESULT_DIR \
    --early_stopping $EARLY_STOPPING \
    --max_input_length 444 \
    --model_type $MODEL_URL \
    --upstream_num_shots $RETRIEVE_NUM \
    --retriever_mode $RETRIEVER \
    --train_seed $TRAIN_SEED \
    --retrieve_seed $RETRIEVE_SEED \
    --save_checkpoint $SAVE_CHECKPOINT \
    --test_on_the_fly True \
    --retrieved_data_dir $RETRIEVED_DATA_DIR

# sbatch scripts/zs_retrieval/zeroshot_random_tune_one.sh 42