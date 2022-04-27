#!/bin/bash
#SBATCH --time=5:00:00
#SBATCH --job-name=retrieve_with_bart
#SBATCH --output jobs/%j.log
#SBATCH --gres=gpu:6000:1

python metax/reranker_bootstrap/bart_retrieve.py \
    --query_path data/bootstrap_reranker/bart_queries/bart_queries_300.json \
    --cache_path memory_cache/bart0_memory.pkl \
    --N 60 \
    --shards 6 \
    --output_path data/bootstrap_reranker/bart_retrieved/ \