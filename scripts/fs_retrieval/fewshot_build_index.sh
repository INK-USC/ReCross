#!/bin/bash
#SBATCH --time=10:00:00
#SBATCH --job-name=metax_build_index
#SBATCH --output jobs/%j.out
#SBATCH --gres=gpu:6000:1
#SBATCH --exclude ink-molly

TYPE=$1

num_shards=$2
shard_id=$3

RETRIEVE_NUM=1000
RETRIEVE_SEED=1111
# Batch size for building the index

QUERY_ENCODER_PATH=""
MEMORY_ENCODER_PATH=""

CKPT_DIR="checkpoints"
export TOKENIZERS_PARALLELISM=false

if [[ "$TYPE" == "Semantic"* ]]; then
    # SentenceBERT index
    retriever_mode="SentenceTransformer"
    cache_path="memory_cache/sbert_memory_fewshot.pkl.${shard_id}_of_${num_shards}"
    LOG_DIR="logs/bart0-fewshot-semantic-cache-index"
    RESULT_DIR="results/bart0-fewshot-semantic-cache-index"
    run_name="build-fewshot-semantic-index-num-${RETRIEVE_NUM}-seed-${RETRIEVE_SEED}.${shard_id}_of_${num_shards}"
    BATCH_SIZE=150
elif [[ "$TYPE" == "BART"* ]]; then
    # BART index
    retriever_mode="bart"
    cache_path="memory_cache/bart0_memory_fewshot.pkl.${shard_id}_of_${num_shards}"
    LOG_DIR="logs/bart0-fewshot-bart-cache-index"
    RESULT_DIR="results/bart0-fewshot-bart-cache-index"
    run_name="build-fewshot-bart-index-num-${RETRIEVE_NUM}-seed-${RETRIEVE_SEED}.${shard_id}_of_${num_shards}"
    # n.b. -- requires a much smaller batch size than SentenceBERT
    BATCH_SIZE=4
elif [[ "$TYPE" == "Trained"* ]]; then
    # BART index
    retriever_mode="trained"
    cache_path="memory_cache/trained_memory_fewshot.pkl.${shard_id}_of_${num_shards}"
    LOG_DIR="logs/trained-fewshot-bart-cache-index"
    RESULT_DIR="results/trained-fewshot-bart-cache-index"
    run_name="build-fewshot-trained-index-num-${RETRIEVE_NUM}-seed-${RETRIEVE_SEED}.${shard_id}_of_${num_shards}"
    # n.b. -- requires a smaller batch size than SentenceBERT
    BATCH_SIZE=8
    QUERY_ENCODER_PATH='outputs/query_encoder_epoch_8.pt'
    MEMORY_ENCODER_PATH='outputs/memory_encoder_epoch_8.pt'
fi

mkdir -p $LOG_DIR
mkdir -p $RESULT_DIR
mkdir -p $CKPT_DIR


# for shard_id in {0..7};
# do
python -m metax.run \
    --run_name $run_name \
    --use_retriever \
    --action=few_shot_build_index_cache \
    --upstream_train_file "data/bart0_upstream_train_lines.json" \
    --num_shards_for_indexing $num_shards --shard_id_for_indexing $shard_id \
    --memory_cache_path ${cache_path} \
    --log_dir=$LOG_DIR \
    --checkpoint=$CKPT_DIR \
    --output_dir=$RESULT_DIR \
    --model_type='yuchenlin/BART0' \
    --upstream_num_shots=$RETRIEVE_NUM \
    --use_retriever \
    --predict_batch_size $BATCH_SIZE \
    --retrieve_seed $RETRIEVE_SEED \
    --retriever_mode $retriever_mode \
    --query_encoder_path $QUERY_ENCODER_PATH \
    --memory_encoder_path $MEMORY_ENCODER_PATH
# done



### sbatch scripts/zs_retrieval/zeroshot_build_index.sh Semantic 0
: '

for shard_id in {0..7};
do
    sbatch scripts/fs_retrieval/fewshot_build_index.sh BART 8 $shard_id
done


python scripts/zs_retrieval/merge_memory_index.py
'
