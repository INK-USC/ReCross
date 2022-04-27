#!/bin/bash
#SBATCH --time=2:00:00
#SBATCH --job-name=metax_build_index
#SBATCH --output jobs/%j.out
#SBATCH --gres=gpu:1
#SBATCH --qos=general
#sSBATCH --exclude ink-titan

TYPE=$1

num_shards=$2
shard_id=$3

RETRIEVE_NUM=1000
RETRIEVE_SEED=1111
# Batch size for building the index
BATCH_SIZE=20

CKPT_DIR="checkpoints"
export TOKENIZERS_PARALLELISM=false
QUERY_ENCODER_PATH="n/a"
MEMORY_ENCODER_PATH="n/a"


if [[ "$TYPE" == "Semantic"* ]]; then
    # SentenceBERT index
    retriever_mode="SentenceTransformer"
    cache_path="memory_cache/sbert_memory.pkl.${shard_id}_of_${num_shards}"
    LOG_DIR="logs/bart0-zeroshot-semantic-cache-index"
    RESULT_DIR="results/bart0-zeroshot-semantic-cache-index"
    run_name="build-zeroshot-semantic-index.${shard_id}_of_${num_shards}"
elif [[ "$TYPE" == "BART"* ]]; then
    # BART index
    retriever_mode="bart"
    cache_path="memory_cache/bart_memory.pkl.${shard_id}_of_${num_shards}"
    LOG_DIR="logs/bart0-zeroshot-bart-cache-index"
    RESULT_DIR="results/bart0-zeroshot-bart-cache-index"
    run_name="build-zeroshot-bart-index.${shard_id}_of_${num_shards}"
    model_type='facebook/bart-large'
elif [[ "$TYPE" == "BART0"* ]]; then
    # BART index
    retriever_mode="bart"
    cache_path="memory_cache/bart0_memory.pkl.${shard_id}_of_${num_shards}"
    LOG_DIR="logs/bart0-zeroshot-bart-cache-index"
    RESULT_DIR="results/bart0-zeroshot-bart-cache-index"
    run_name="build-zeroshot-bart-index.${shard_id}_of_${num_shards}"
    model_type='yuchenlin/BART0'
elif [[ "$TYPE" == "Trained"* ]]; then
    # biencoder index
    retriever_mode="trained"
    cache_path="memory_cache/trained_memory.pkl.${shard_id}_of_${num_shards}"
    LOG_DIR="logs/trained-bart-cache-index"
    RESULT_DIR="results/trained-bart-cache-index"
    run_name="build-trained-index-num-${RETRIEVE_NUM}-seed-${RETRIEVE_SEED}.${shard_id}_of_${num_shards}"
    # n.b. -- requires a smaller batch size than SentenceBERT
    model_type='yuchenlin/BART0'
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
    --action=zero_shot_build_index_cache \
    --upstream_train_file "data/bart0_upstream_train_lines.json" \
    --num_shards_for_indexing $num_shards --shard_id_for_indexing $shard_id \
    --memory_cache_path ${cache_path} \
    --log_dir=$LOG_DIR \
    --checkpoint=$CKPT_DIR \
    --output_dir=$RESULT_DIR \
    --model_type=${model_type} \
    --ret_merge_mode 'unsupervised' \
    --max_input_length 512 \
    --predict_batch_size $BATCH_SIZE \
    --query_encoder_path ${QUERY_ENCODER_PATH} \
    --memory_encoder_path ${MEMORY_ENCODER_PATH} \
    --retriever_mode $retriever_mode
# done



### sbatch scripts/ret/build_index.sh Semantic 0
: '

for shard_id in {0..15};
do
    sbatch scripts/ret/build_index.sh Semantic 16 $shard_id
done

# sbatch scripts/ret/build_index.sh Semantic 16 0

python scripts/ret/merge_memory_index.py memory_cache/sbert_memory.pkl 16

----

for shard_id in {0..31};
do
    sbatch scripts/ret/build_index.sh BART0 32 $shard_id
done

python scripts/ret/merge_memory_index.py memory_cache/bart0_memory.pkl 32



'