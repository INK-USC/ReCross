#!/bin/bash
#SBATCH --time=5:00:00
#SBATCH --job-name=fs_random_all
#SBATCH --output jobs/%j.out
#SBATCH --gres=gpu:8000:1 
#SBATCH --qos=general-8000
#sSBATCH --gres=gpu:6000:1 
#sSBATCH --qos=general


EARLY_STOPPING=True
# TARGET_TASK=$1
LR=1e-5
RETLR=6e-6
EPOCH=2     # 1st stage 
FS_EPOCH=5 # 2nd stage  
MODEL="yuchenlin/BART0"
MODE="two-stage" #  MODE="two-stage"
RETRIEVER=$2
RETRIEVE_SEED=$3    # 42, 43, ...
finetune_round=$4
RERANK=$5

# declare -a RETRIEVERS=("Random" "SentenceTransformer" "BART")
# declare -a RETRIEVERS=("Random")
# for RETRIEVER in "${RETRIEVERS[@]}" 
# do
SHOTS=32
RETRIEVE_NUM=512
LOG_DIR="logs/bart0-fewshot_${RETRIEVER}"
RESULT_DIR="results/bart0-fewshot_${RETRIEVER}"
CKPT_DIR="checkpoints"
RETRIEVED_DATA_DIR="retrieved_data/fs_${RETRIEVER}" # dir to save data retrieved by retrievers
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

if [[ "$RERANK" == "rerank"* ]]; then
    reranker_model_path="checkpoints/reranker/roberta_reranker_v3/checkpoint-9000"
    reranker_oversample_rate=3
    RETRIEVED_DATA_DIR="retrieved_data/fs_${RETRIEVER}_reranked" # dir to save data retrieved by retrievers
else  
    reranker_model_path=" "
    reranker_oversample_rate=1
    RETRIEVED_DATA_DIR="retrieved_data/fs_${RETRIEVER}" # dir to save data retrieved by retrievers
fi

mkdir -p $RETRIEVED_DATA_DIR


TASKS_IN=$1

tasks=$(echo $TASKS_IN | tr "," "\n")

for task in $tasks
do
    echo "> [$task]"
    
    TARGET_TASK=$task
    run_name="RET_${RETRIEVER}-${MODE}_${RERANK}-${SHOTS}-${RETRIEVE_NUM}-${RETRIEVE_SEED}-${TARGET_TASK}-${RETLR}-${EPOCH}"
    echo ${run_name}

    python -m metax.run \
        --run_name "${run_name}" \
        --use_retriever --retriever_mode $RETRIEVER \
        --ret_merge_mode $MODE \
        --memory_cache_path $memory_cache_path \
        --action "ret_aug" \
        --query_aggregation_mode "aggregate_choices" \
        --log_dir $LOG_DIR \
        --checkpoint $CKPT_DIR \
        --num_train_epochs ${EPOCH} \
        --num_train_epochs_fs ${FS_EPOCH} \
        --num_shots $SHOTS \
        --warmup_steps 0 \
        --learning_rate $LR \
        --ret_learning_rate $RETLR \
        --train_batch_size 4 \
        --gradient_accumulation_steps 1 \
        --predict_batch_size 8 \
        --target_task $TARGET_TASK \
        --output_dir $RESULT_DIR \
        --early_stopping $EARLY_STOPPING \
        --max_input_length 386 \
        --max_input_length_for_eval 512 \
        --model_type ${MODEL} \
        --retrieve_seed $RETRIEVE_SEED \
        --test_on_the_fly True \
        --upstream_num_shots ${RETRIEVE_NUM} \
        --reranker_model_path "${reranker_model_path}" \
        --reranker_oversample_rate ${reranker_oversample_rate} \
        --retrieved_data_dir ${RETRIEVED_DATA_DIR} \
        --finetune_round ${finetune_round} &
done
wait;


: '
# declare -a seeds=(1337 42 1213 2022 2333)
# # declare -a seeds=(1338 43 1214 2023 2334)
# for seed in "${seeds[@]}" 
# do
#     sbatch scripts/ret/zs_with_ret.sh all Random $seed 1
# done


data_names=($(cat scripts/full_downstream.tasks))
for data_name in "${data_names[@]}" 
do
    sbatch scripts/ret/fs_with_ret.sh $data_name Random 42 10
done



sbatch scripts/ret/fs_with_ret.sh  ai2_arc-ARC-Easy,ai2_arc-ARC-Challenge  Random 42 5 no
sbatch scripts/ret/fs_with_ret.sh  squad_v2,hellaswag Random 42 5 no
sbatch scripts/ret/fs_with_ret.sh  openbookqa-main,super_glue-multirc  Random 42 5 no
sbatch scripts/ret/fs_with_ret.sh  super_glue-boolq,super_glue-wic  Random 42 5 no
sbatch scripts/ret/fs_with_ret.sh  super_glue-wsc.fixed,winogrande-winogrande_xl  Random 42 5 no
sbatch scripts/ret/fs_with_ret.sh  super_glue-cb,super_glue-rte  Random 42 5 no
sbatch scripts/ret/fs_with_ret.sh  anli_r1,anli_r2  Random 42 5 no
sbatch scripts/ret/fs_with_ret.sh  anli_r3,piqa Random 42 5 no




sbatch scripts/ret/fs_with_ret.sh  ai2_arc-ARC-Easy,ai2_arc-ARC-Challenge  Random 42 5 rerank
sbatch scripts/ret/fs_with_ret.sh  squad_v2,hellaswag Random 42 5 rerank
sbatch scripts/ret/fs_with_ret.sh  openbookqa-main,super_glue-multirc  Random 42 5 rerank
sbatch scripts/ret/fs_with_ret.sh  super_glue-boolq,super_glue-wic  Random 42 5 rerank
sbatch scripts/ret/fs_with_ret.sh  super_glue-wsc.fixed,winogrande-winogrande_xl  Random 42 5 rerank
sbatch scripts/ret/fs_with_ret.sh  super_glue-cb,super_glue-rte  Random 42 5 rerank
sbatch scripts/ret/fs_with_ret.sh  anli_r1,anli_r2  Random 42 5 rerank
sbatch scripts/ret/fs_with_ret.sh  anli_r3,piqa Random 42 5 rerank



sbatch scripts/ret/fs_with_ret.sh  ai2_arc-ARC-Easy,ai2_arc-ARC-Challenge  SentenceTransformer 42 10 rerank
sbatch scripts/ret/fs_with_ret.sh  squad_v2,hellaswag SentenceTransformer 42 10 rerank
sbatch scripts/ret/fs_with_ret.sh  openbookqa-main,super_glue-multirc  SentenceTransformer 42 10 rerank
sbatch scripts/ret/fs_with_ret.sh  super_glue-boolq,super_glue-wic  SentenceTransformer 42 10 rerank
sbatch scripts/ret/fs_with_ret.sh  super_glue-wsc.fixed,winogrande-winogrande_xl  SentenceTransformer 42 10 rerank
sbatch scripts/ret/fs_with_ret.sh  super_glue-cb,super_glue-rte  SentenceTransformer 42 10 rerank
sbatch scripts/ret/fs_with_ret.sh  anli_r1,anli_r2  SentenceTransformer 42 10 rerank
sbatch scripts/ret/fs_with_ret.sh  anli_r3,piqa SentenceTransformer 42 10 rerank

 
'