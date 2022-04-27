#!/bin/bash
#SBATCH --time=1:00:00
#SBATCH --job-name=metax_build_index
#SBATCH --output jobs/%j.out
#SBATCH --gres=gpu:1
#SBATCH --qos=general
#SBATCH --exclude ink-ron

TYPE=$1

num_shards=$2
shard_id=$3

RETRIEVE_NUM=1000
RETRIEVE_SEED=1111
# Batch size for building the index
BATCH_SIZE=4

CKPT_DIR="checkpoints"
export TOKENIZERS_PARALLELISM=false

if [[ "$TYPE" == "Semantic"* ]]; then
    # SentenceBERT index
    retriever_mode="SentenceTransformer"
    cache_path="memory_cache/csr_sbert_memory.pkl.${shard_id}_of_${num_shards}"
    LOG_DIR="logs/csr-zeroshot-semantic-cache-index"
    RESULT_DIR="results/csr-zeroshot-semantic-cache-index"
    run_name="build-zeroshot-semantic-index.${shard_id}_of_${num_shards}"
elif [[ "$TYPE" == "BART"* ]]; then
    # BART index
    retriever_mode="bart"
    cache_path="memory_cache/csr_bart_memory.pkl.${shard_id}_of_${num_shards}"
    LOG_DIR="logs/csr-zeroshot-bart-cache-index"
    RESULT_DIR="results/csr-zeroshot-bart-cache-index"
    run_name="build-zeroshot-bart-index.${shard_id}_of_${num_shards}"
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
    --upstream_train_file "data/csr_upstream_train_lines.json" \
    --num_shards_for_indexing $num_shards --shard_id_for_indexing $shard_id \
    --memory_cache_path ${cache_path} \
    --log_dir=$LOG_DIR \
    --checkpoint=$CKPT_DIR \
    --output_dir=$RESULT_DIR \
    --model_type='facebook/bart-large' \
    --max_input_length 444 \
    --predict_batch_size $BATCH_SIZE \
    --retriever_mode $retriever_mode
# done



### sbatch scripts/csr_scripts/build_csr_index.sh Semantic 0
: '

for shard_id in {0..7};
do
    sbatch scripts/csr_scripts/build_csr_index.sh Semantic 8 $shard_id
done

python scripts/zs_retrieval/merge_memory_index.py memory_cache/csr_sbert_memory.pkl 8

for shard_id in {0..19};
do
    sbatch scripts/csr_scripts/build_csr_index.sh BART 20 $shard_id
done

python scripts/zs_retrieval/merge_memory_index.py memory_cache/csr_bart_memory.pkl 20

'