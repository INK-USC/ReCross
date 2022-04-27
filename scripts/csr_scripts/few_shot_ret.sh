#!/bin/bash
#SBATCH --time=5:00:00
#SBATCH --job-name=fs_random_all
#SBATCH --output jobs/%j.out



#SBATCH --gres=gpu:1 
#SBATCH --qos=general-8000

#sSBATCH --gres=gpu:6000:1 
#sSBATCH --qos=general


EARLY_STOPPING=True
# TARGET_TASK=$1
LR=1e-5
RETLR=1e-5
EPOCH=3     # 1st stage 
FS_EPOCH=5 # 2nd stage  
# MODEL="yuchenlin/BART0"

MODEL="yuchenlin/BART0_CSR"
MODEL_STR="BART0_CSR"
# MODEL="facebook/bart-large"
# MODEL_STR="BART"


MODE="two-stage" #  MODE="two-stage"
RETRIEVER=$2
RETRIEVE_SEED=$3    # 42, 43, ...
finetune_round=$4

# declare -a RETRIEVERS=("Random" "SentenceTransformer" "BART")
# declare -a RETRIEVERS=("Random")
# for RETRIEVER in "${RETRIEVERS[@]}" 
# do
SHOTS=64
RETRIEVE_NUM=512
LOG_DIR="logs/csr-fewshot_${RETRIEVER}"
RESULT_DIR="results/csr-fewshot_${RETRIEVER}"
CKPT_DIR="checkpoints"
RETRIEVED_DATA_DIR="retrieved_data/csr_fs_${RETRIEVER}" # dir to save data retrieved by retrievers
mkdir -p $LOG_DIR
mkdir -p $RESULT_DIR
mkdir -p $CKPT_DIR
mkdir -p $RETRIEVED_DATA_DIR


if [[ "$RETRIEVER" == "SentenceTransformer"* ]]; then
    memory_cache_path="memory_cache/csr_sbert_memory.pkl"
elif [[ "$RETRIEVER" == "Random"* ]]; then 
    memory_cache_path="memory_cache/csr_random_memory.pkl"
elif [[ "$RETRIEVER" == "BART"* ]]; then 
    memory_cache_path="memory_cache/csr_bart_memory.pkl"
fi


TASKS_IN=$1

tasks=$(echo $TASKS_IN | tr "," "\n")

for task in $tasks
do
    echo "> [$task]"
    
    TARGET_TASK=$task
    run_name="RET_${MODEL_STR}_${RETRIEVER}-${MODE}-${SHOTS}-${RETRIEVE_NUM}-${RETRIEVE_SEED}-${TARGET_TASK}-${RETLR}-${EPOCH}"
    echo ${run_name}

    python -m metax.run \
        --run_name "${run_name}" \
        --upstream_train_file "data/csr_upstream_train_lines.json" \
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
        --train_batch_size 6 \
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


data_names=($(cat scripts/csr_downstream.tasks))
for data_name in "${data_names[@]}" 
do
    sbatch scripts/ret/fs_with_ret.sh $data_name Random 42 5
done



# sbatch scripts/ret/fs_with_ret.sh  ai2_arc-ARC-Easy,ai2_arc-ARC-Challenge  Random 42 10
# sbatch scripts/ret/fs_with_ret.sh  squad_v2,hellaswag Random 42 10
# sbatch scripts/ret/fs_with_ret.sh  openbookqa-main,super_glue-multirc  Random 42 10
# sbatch scripts/ret/fs_with_ret.sh  super_glue-boolq,super_glue-wic  Random 42 10
# sbatch scripts/ret/fs_with_ret.sh  super_glue-wsc.fixed,winogrande-winogrande_xl  Random 42 10
# sbatch scripts/ret/fs_with_ret.sh  super_glue-cb,super_glue-rte  Random 42 10
# sbatch scripts/ret/fs_with_ret.sh  anli_r1,anli_r2  Random 42 10
# sbatch scripts/ret/fs_with_ret.sh  anli_r3,piqa Random 42 10

sbatch scripts/csr_scripts/few_shot_ret.sh  ag_news,rotten_tomatoes  Random 42 5
sbatch scripts/csr_scripts/few_shot_ret.sh  glue-mrpc,glue-qqp  Random 42 5
sbatch scripts/csr_scripts/few_shot_ret.sh  imdb,squad_v2  Random 42 5
sbatch scripts/csr_scripts/few_shot_ret.sh  super_glue-boolq,super_glue-wic  Random 42 5
sbatch scripts/csr_scripts/few_shot_ret.sh  super_glue-cb,super_glue-rte  Random 42 5


sbatch scripts/csr_scripts/few_shot_ret.sh  ag_news,rotten_tomatoes  SentenceTransformer 42 5
sbatch scripts/csr_scripts/few_shot_ret.sh  glue-mrpc,glue-qqp  SentenceTransformer 42 5
sbatch scripts/csr_scripts/few_shot_ret.sh  imdb,squad_v2  SentenceTransformer 42 5
sbatch scripts/csr_scripts/few_shot_ret.sh  super_glue-boolq,super_glue-wic  SentenceTransformer 42 5
sbatch scripts/csr_scripts/few_shot_ret.sh  super_glue-cb,super_glue-rte  SentenceTransformer 42 5


sbatch scripts/csr_scripts/few_shot_ret.sh  ag_news,rotten_tomatoes  BART 42 5
sbatch scripts/csr_scripts/few_shot_ret.sh  glue-mrpc,glue-qqp  BART 42 5
sbatch scripts/csr_scripts/few_shot_ret.sh  imdb,squad_v2  BART 42 5
sbatch scripts/csr_scripts/few_shot_ret.sh  super_glue-boolq,super_glue-wic  BART 42 5
sbatch scripts/csr_scripts/few_shot_ret.sh  super_glue-cb,super_glue-rte  BART 42 5



sbatch scripts/csr_scripts/few_shot_ret.sh  squad_v2  SentenceTransformer 42 5
sbatch scripts/csr_scripts/few_shot_ret.sh  squad_v2  BART 42 5
'