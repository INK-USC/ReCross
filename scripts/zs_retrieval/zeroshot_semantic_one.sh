#!/bin/bash
#SBATCH --time=1:00:00
#SBATCH --job-name=metax-zs-sbert
#SBATCH --output jobs/%j.out
#SBATCH --gres=gpu:6000:1
#SBATCH --exclude ink-molly
RETRIEVE_NUM=500
EARLY_STOPPING=False
TRAIN_SEED=3331
MODEL_NAME="BART0"
TARGET_TASK=$1
RETRIEVE_SEED=$2
LR=3e-6 
EPOCH=3
TOTAL_STEPS=200
RETRIEVER="SentenceTransformer"
MODEL_URL="yuchenlin/BART0"
LOG_DIR="logs/zeroshot_${RETRIEVER}"
RETRIEVED_DATA_DIR="retrieved_data/zeroshot_${RETRIEVER}/sbert-${TARGET_TASK}" # dir to save data retrieved by retrievers
RESULT_DIR="results/zeroshot_${RETRIEVER}"
CKPT_DIR="checkpoints"
CKPT_PREFIX="${MODEL_NAME}-${RETRIEVER}-${RETRIEVE_SEED}-${TRAIN_SEED}-${LR}"
mkdir -p $LOG_DIR
mkdir -p $RESULT_DIR
mkdir -p $CKPT_DIR
mkdir -p $RETRIEVED_DATA_DIR

export TOKENIZERS_PARALLELISM=false

python -m metax.run \
    --run_name "${RETRIEVER}-${RETRIEVE_NUM}-${RETRIEVE_SEED}-${TARGET_TASK}-${LR}-${TOTAL_STEPS}" \
    --use_retriever --action zs_retrieve_pipeline \
    --upstream_train_file "data/bart0_upstream_train_lines.json" \
    --memory_cache_path "memory_cache/sbert_memory.pkl" \
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
    --model_type ${MODEL_URL} \
    --upstream_num_shots $RETRIEVE_NUM \
    --train_seed $TRAIN_SEED \
    --retrieve_seed $RETRIEVE_SEED \
    --retriever_mode $RETRIEVER \
    --retrieved_data_dir $RETRIEVED_DATA_DIR \
    --test_on_the_fly True \
    --query_aggregation_mode "aggregate_choices"


# sbatch scripts/zs_retrieval/zeroshot_semantic_one.sh anli_r1 1337
# sbatch scripts/zs_retrieval/zeroshot_semantic_one.sh anli_r1 2022
# sbatch scripts/zs_retrieval/zeroshot_semantic_one.sh anli_r2 42
# sbatch scripts/zs_retrieval/zeroshot_semantic_one.sh story_cloze-2016 1337
# sbatch scripts/zs_retrieval/zeroshot_semantic_one.sh super_glue-cb 1337
# sbatch scripts/zs_retrieval/zeroshot_semantic_one.sh super_glue-cb 1213