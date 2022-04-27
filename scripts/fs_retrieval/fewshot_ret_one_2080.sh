#!/bin/bash
#SBATCH --time=3:00:00
#SBATCH --job-name=metax_fs_eval
#SBATCH --output jobs/%j.out
#SBATCH --gres=gpu:1 
#SBATCH --qos=general
#SBATCH --exclude ink-ron

EARLY_STOPPING=False
TARGET_TASK=$1
LR=1e-5
RETLR=3e-6
EPOCH=3
MODEL="yuchenlin/BART0"
MODE=$2 # "two-stage"

RETRIEVER=$3 # "Random"



# declare -a RETRIEVERS=("Random" "SentenceTransformer" "BART")
# declare -a RETRIEVERS=("Random")
# for RETRIEVER in "${RETRIEVERS[@]}" 
# do
SHOTS=64
RETRIEVE_NUM=512
LOG_DIR="logs/bart0-fewshot_${RETRIEVER}_v2"
RESULT_DIR="results/bart0-fewshot_${RETRIEVER}_v2"
CKPT_DIR="checkpoints"
mkdir -p $LOG_DIR
mkdir -p $RESULT_DIR
mkdir -p $CKPT_DIR


if [[ "$RETRIEVER" == "SentenceTransformer"* ]]; then
    memory_cache_path="memory_cache/sbert_memory.pkl"
elif [[ "$RETRIEVER" == "Random"* ]]; then 
    memory_cache_path="memory_cache/random_memory.pkl"
elif [[ "$RETRIEVER" == "BART"* ]]; then 
    memory_cache_path="memory_cache/bart0_memory.pkl"
fi

echo $RETRIEVER
run_name="FS_${RETRIEVER}-${MODE}-${SHOTS}-${RETRIEVE_NUM}-${EARLY_STOPPING}-${TARGET_TASK}-${LR}-${EPOCH}"
echo ${run_name}

python -m metax.run \
    --run_name "${run_name}" \
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
    --train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --predict_batch_size 4 \
    --target_task $TARGET_TASK \
    --output_dir $RESULT_DIR \
    --early_stopping $EARLY_STOPPING \
    --max_input_length 256 \
    --max_input_length_for_eval 512 \
    --model_type ${MODEL} \
    --test_on_the_fly True \
    --upstream_num_shots ${RETRIEVE_NUM} \
    --finetune_round 3 
# done

# wait;

: '

# declare -a data_names=("winogrande-winogrande_xl" "super_glue-cb" "super_glue-rte" "anli_r2" "anli_r3" "story_cloze-2016" "super_glue-wsc.fixed" "hellaswag" "super_glue-copa" "super_glue-wic")

data_names=($(cat scripts/all_downstream.tasks))
declare -a RETRIEVERS=("Random" "SentenceTransformer" "BART")
for data_name in "${data_names[@]}" 
do
for ret in in "${RETRIEVERS[@]}" 
do
    sbatch scripts/fs_retrieval/fewshot_ret_one_2080.sh ${data_name} two-stage ${ret}
    echo "Submitted: ${data_name} w/ ${ret}"
done
done  
'

sbatch scripts/fs_retrieval/fewshot_ret_one_2080.sh anli_r2 two-stage Random



    # \
    # --finetune_layers=1

# sbatch scripts/fs_retrieval/fewshot_ret_one.sh super_glue-cb mix
# sbatch scripts/fs_retrieval/fewshot_ret_one.sh super_glue-cb SentenceTransformer mix

# sbatch scripts/fs_retrieval/fewshot_ret_one.sh super_glue-cb Random two-stage
# sbatch scripts/fs_retrieval/fewshot_ret_one.sh super_glue-cb SentenceTransformer two-stage
# sbatch scripts/fs_retrieval/fewshot_ret_one.sh super_glue-cb BART two-stage

# CUDA_VISIBLE_DEVICES=0 bash scripts/fs_retrieval/fewshot_ret_one.sh super_glue-cb  Random two-stage
# CUDA_VISIBLE_DEVICES=0 bash scripts/fs_retrieval/fewshot_ret_one.sh super_glue-cb  BART mix

