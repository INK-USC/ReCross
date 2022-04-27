#!/bin/bash
#SBATCH --time=5:00:00
#SBATCH --job-name=zs_random_all
#SBATCH --output jobs/%j.out
#SBATCH --gres=gpu:8000:1
#SBATCH --qos=general-8000
#sSBATCH --gres=gpu:6000:1
#sSBATCH --qos=general
#sSBATCH --exclude ink-nova

EARLY_STOPPING=False
# TARGET_TASK=$1
TASKS_IN=$1
LR=1e-5
RETLR=6e-6
EPOCH=2
Train_batch_size=4
MODEL="yuchenlin/BART0"

# MODEL="bigscience/T0_3B"
# Train_batch_size=2

MODE="unsupervised"
RETRIEVER=$2
SEEDS_IN=$3
# RETRIEVE_SEED=$3    # 42, 43, ...
finetune_round=$4
RERANK=$5
# REMOVE_GROUP=$6
REMOVE_GROUP="none"

# declare -a RETRIEVERS=("Random" "SentenceTransformer" "BART")
# declare -a RETRIEVERS=("Random")
# for RETRIEVER in "${RETRIEVERS[@]}"
# do
SHOTS=16
RETRIEVE_NUM=512
LOG_DIR="logs/bart0-zeroshot_${RETRIEVER}"
RESULT_DIR="results/bart0-zeroshot_${RETRIEVER}"
CKPT_DIR="checkpoints"
query_aggregation_mode="choices"

mkdir -p $LOG_DIR
mkdir -p $RESULT_DIR
mkdir -p $CKPT_DIR

QUERY_ENCODER_PATH="/"
MEMORY_ENCODER_PATH="/"

RERANKER_NAME="none"
if [[ "$RERANK" == "rerank"* ]]; then
    # reranker_model_path="checkpoints/reranker/bs_from_scratch/checkpoint-600"

    # reranker_model_path="/home/kangmin/MetaCross/checkpoints/reranker/BART0_base/iteration0/checkpoint-510-epoch-1"
    # RERANKER_NAME="bart0_base"
    
    # reranker_model_path="/home/bill/MetaCross/checkpoints/reranker/BART0_large/iteration0/checkpoint-1200"
    # RERANKER_NAME="bart0"

    reranker_model_path="/home/kangmin/MetaCross/checkpoints/reranker/roberta_base/iteration0/checkpoint-600"
    RERANKER_NAME="roberta_base"

    # reranker_model_path="/home/kangmin/MetaCross/checkpoints/reranker/roberta_base/iteration1/checkpoint-800"
    # RERANKER_NAME="roberta_baseI2"

    # reranker_model_path="/home/bill/MetaCross/checkpoints/reranker/roberta_base/train_from_scratch_iter0_1_merged/checkpoint-1400"
    # RERANKER_NAME="roberta_baseI01m"

    # reranker_model_path="/home/bill/MetaCross/checkpoints/reranker/roberta_base/iteration1_train_from_scratch/checkpoint-800/"
    # RERANKER_NAME="roberta_baseI1"    

    reranker_oversample_rate=4
    reranker_query_size=-1
    RETRIEVED_DATA_DIR="retrieved_data/zs_${RETRIEVER}_reranked" # dir to save data retrieved by retrievers
else
    reranker_model_path=" "
    reranker_oversample_rate=1
    reranker_query_size=-1
    RETRIEVED_DATA_DIR="retrieved_data/zs_${RETRIEVER}" # dir to save data retrieved by retrievers
fi


if [[ "$RETRIEVER" == "SentenceTransformer"* ]]; then
    memory_cache_path="memory_cache/sbert_memory.pkl"
elif [[ "$RETRIEVER" == "Random"* ]]; then
    memory_cache_path="memory_cache/random_memory.pkl"
elif [[ "$RETRIEVER" == "BART"* ]]; then
    memory_cache_path="memory_cache/bart0_memory.pkl" 
fi

mkdir -p $RETRIEVED_DATA_DIR

seeds=$(echo $SEEDS_IN | tr "," "\n")
tasks=$(echo $TASKS_IN | tr "," "\n")

for seed in $seeds
do

RETRIEVE_SEED=$seed

for task in $tasks
do
TARGET_TASK=$task
run_name="RET_${RETRIEVER}-${MODE}_${RERANK}_${RERANKER_NAME}-${reranker_oversample_rate}-${SHOTS}-${RETRIEVE_NUM}-${RETRIEVE_SEED}-${TARGET_TASK}-${RETLR}-${EPOCH}"
# run_name="RET_${RETRIEVER}-${MODE}_${RERANK}_${RERANKER_NAME}-${REMOVE_GROUP}-${reranker_oversample_rate}-${SHOTS}-${RETRIEVE_NUM}-${RETRIEVE_SEED}-${TARGET_TASK}-${RETLR}-${EPOCH}"
echo "> [$task]"
echo $RETRIEVER
echo ${run_name}

python -m metax.run \
    --run_name "${run_name}" \
    --use_retriever --retriever_mode $RETRIEVER \
    --query_encoder_path ${QUERY_ENCODER_PATH} \
    --memory_encoder_path ${MEMORY_ENCODER_PATH} \
    --ret_merge_mode $MODE \
    --memory_cache_path $memory_cache_path \
    --action "ret_aug" \
    --query_aggregation_mode "aggregate_${query_aggregation_mode}" \
    --log_dir $LOG_DIR \
    --checkpoint $CKPT_DIR \
    --num_train_epochs ${EPOCH} \
    --num_shots $SHOTS \
    --warmup_steps 0 \
    --learning_rate $LR \
    --ret_learning_rate $RETLR \
    --train_batch_size ${Train_batch_size} \
    --gradient_accumulation_steps 1 \
    --predict_batch_size 8 \
    --target_task $TARGET_TASK \
    --remove_group $REMOVE_GROUP \
    --output_dir $RESULT_DIR \
    --early_stopping $EARLY_STOPPING \
    --max_input_length 512 \
    --max_input_length_for_eval 512 \
    --model_type ${MODEL} \
    --retrieve_seed $RETRIEVE_SEED \
    --test_on_the_fly True \
    --upstream_num_shots ${RETRIEVE_NUM} \
    --reranker_model_path "${reranker_model_path}" \
    --reranker_oversample_rate ${reranker_oversample_rate} \
    --reranker_query_size ${reranker_query_size} \
    --retrieved_data_dir ${RETRIEVED_DATA_DIR} \
    --finetune_round ${finetune_round}  &

done
done
wait;


: '
declare -a seeds=(1337 1213 2022 2333 1338 43 1214 2023 2334)
for seed in "${seeds[@]}"
do
    sbatch scripts/ret/zs_with_ret.sh all Random $seed 1
done


sbatch scripts/ret/zs_with_ret.sh all Random 1337,1338 1 no
sbatch scripts/ret/zs_with_ret.sh all Random 42,43 1 no
sbatch scripts/ret/zs_with_ret.sh all Random 1213,1214 1 no
sbatch scripts/ret/zs_with_ret.sh all Random 2022,2023 1 no
sbatch scripts/ret/zs_with_ret.sh all Random 2333,2334 1 no



# sbatch scripts/ret/zs_with_ret.sh all Random 1337,1338 1 rerank
# sbatch scripts/ret/zs_with_ret.sh all Random 42,43 1 rerank
# sbatch scripts/ret/zs_with_ret.sh all Random 1213,1214 1 rerank
# sbatch scripts/ret/zs_with_ret.sh all Random 2022,2023 1 rerank
# sbatch scripts/ret/zs_with_ret.sh all Random 2333,2334 1 rerank






sbatch scripts/ret/zs_with_ret.sh  ai2_arc-ARC-Challenge,super_glue-cb  Random 42 5 no
sbatch scripts/ret/zs_with_ret.sh  squad_v2,hellaswag Random 42 5 no
sbatch scripts/ret/zs_with_ret.sh  openbookqa-main,super_glue-wic  Random 42 5 no
sbatch scripts/ret/zs_with_ret.sh  super_glue-wsc.fixed,winogrande-winogrande_xl  Random 42 5 no
sbatch scripts/ret/zs_with_ret.sh  anli_r3,piqa Random 42 5 no
# sbatch scripts/ret/zs_with_ret.sh  anli_r1,anli_r2  Random 42 5 no


sbatch scripts/ret/zs_with_ret.sh  ai2_arc-ARC-Challenge,super_glue-cb  SentenceTransformer 42 5 no
sbatch scripts/ret/zs_with_ret.sh  squad_v2,hellaswag SentenceTransformer 42 5 no
sbatch scripts/ret/zs_with_ret.sh  openbookqa-main,super_glue-wic  SentenceTransformer 42 5 no
sbatch scripts/ret/zs_with_ret.sh  super_glue-wsc.fixed,winogrande-winogrande_xl  SentenceTransformer 42 5 no
sbatch scripts/ret/zs_with_ret.sh  anli_r3,piqa SentenceTransformer 42 5 no
# sbatch scripts/ret/zs_with_ret.sh  anli_r1,anli_r2  SentenceTransformer 42 5 no


sbatch scripts/ret/zs_with_ret.sh  ai2_arc-ARC-Challenge,super_glue-cb  BART 42 5 no
sbatch scripts/ret/zs_with_ret.sh  squad_v2,hellaswag BART 42 5 no
sbatch scripts/ret/zs_with_ret.sh  openbookqa-main,super_glue-wic  BART 42 5 no
sbatch scripts/ret/zs_with_ret.sh  super_glue-wsc.fixed,winogrande-winogrande_xl  BART 42 5 no
sbatch scripts/ret/zs_with_ret.sh  anli_r3,piqa BART 42 5 no
# sbatch scripts/ret/zs_with_ret.sh  anli_r1,anli_r2  BART 42 5 no

# sbatch scripts/ret/zs_with_ret.sh  openbookqa-main,hellaswag  BART 42 5 no


sbatch scripts/ret/zs_with_ret.sh  ai2_arc-ARC-Challenge,super_glue-cb  Random 42 5 rerank
sbatch scripts/ret/zs_with_ret.sh  squad_v2,hellaswag Random 42 5 rerank
sbatch scripts/ret/zs_with_ret.sh  openbookqa-main,super_glue-wic  Random 42 5 rerank
sbatch scripts/ret/zs_with_ret.sh  super_glue-wsc.fixed,winogrande-winogrande_xl  Random 42 5 rerank
sbatch scripts/ret/zs_with_ret.sh  anli_r3,piqa Random 42 5 rerank
# sbatch scripts/ret/zs_with_ret.sh  anli_r1,anli_r2  Random 42 5 rerank


sbatch scripts/ret/zs_with_ret.sh  ai2_arc-ARC-Challenge,super_glue-cb  SentenceTransformer 42 5 rerank
sbatch scripts/ret/zs_with_ret.sh  squad_v2,hellaswag SentenceTransformer 42 5 rerank
sbatch scripts/ret/zs_with_ret.sh  openbookqa-main,super_glue-wic  SentenceTransformer 42 5 rerank
sbatch scripts/ret/zs_with_ret.sh  super_glue-wsc.fixed,winogrande-winogrande_xl  SentenceTransformer 42 5 rerank
sbatch scripts/ret/zs_with_ret.sh  anli_r3,piqa SentenceTransformer 42 5 rerank
# sbatch scripts/ret/zs_with_ret.sh  anli_r1,anli_r2  SentenceTransformer 42 5 rerank


################ The most imporatnat one!!!!!!!!!

sbatch scripts/ret/zs_with_ret.sh  ai2_arc-ARC-Challenge,super_glue-cb  BART 42 5 rerank
sbatch scripts/ret/zs_with_ret.sh  squad_v2,hellaswag BART 42 5 rerank
sbatch scripts/ret/zs_with_ret.sh  openbookqa-main,super_glue-wic  BART 42 5 rerank
sbatch scripts/ret/zs_with_ret.sh  super_glue-wsc.fixed,winogrande-winogrande_xl  BART 42 5 rerank
sbatch scripts/ret/zs_with_ret.sh  anli_r3,piqa BART 42 5 rerank




# abalation 

sbatch scripts/ret/zs_with_ret.sh  ai2_arc-ARC-Challenge,super_glue-cb  BART 42 5 rerank multiple_choice_qa
sbatch scripts/ret/zs_with_ret.sh  squad_v2,hellaswag BART 42 5 rerank multiple_choice_qa
sbatch scripts/ret/zs_with_ret.sh  openbookqa-main,super_glue-wic  BART 42 5 rerank multiple_choice_qa
sbatch scripts/ret/zs_with_ret.sh  super_glue-wsc.fixed,winogrande-winogrande_xl  BART 42 5 rerank multiple_choice_qa
sbatch scripts/ret/zs_with_ret.sh  anli_r3,piqa BART 42 5 rerank multiple_choice_qa


sbatch scripts/ret/zs_with_ret.sh  ai2_arc-ARC-Challenge,super_glue-cb  BART 42 5 rerank summarization
sbatch scripts/ret/zs_with_ret.sh  squad_v2,hellaswag BART 42 5 rerank summarization
sbatch scripts/ret/zs_with_ret.sh  openbookqa-main,super_glue-wic  BART 42 5 rerank summarization
sbatch scripts/ret/zs_with_ret.sh  super_glue-wsc.fixed,winogrande-winogrande_xl  BART 42 5 rerank summarization
sbatch scripts/ret/zs_with_ret.sh  anli_r3,piqa BART 42 5 rerank summarization

sbatch scripts/ret/zs_with_ret.sh  ai2_arc-ARC-Challenge,super_glue-cb  BART 42 5 rerank extractive_qa
sbatch scripts/ret/zs_with_ret.sh  squad_v2,hellaswag BART 42 5 rerank extractive_qa
sbatch scripts/ret/zs_with_ret.sh  openbookqa-main,super_glue-wic  BART 42 5 rerank extractive_qa
sbatch scripts/ret/zs_with_ret.sh  super_glue-wsc.fixed,winogrande-winogrande_xl  BART 42 5 rerank extractive_qa
sbatch scripts/ret/zs_with_ret.sh  anli_r3,piqa BART 42 5 rerank extractive_qa

sbatch scripts/ret/zs_with_ret.sh  ai2_arc-ARC-Challenge,super_glue-cb  BART 42 5 rerank sentiment
sbatch scripts/ret/zs_with_ret.sh  squad_v2,hellaswag BART 42 5 rerank sentiment
sbatch scripts/ret/zs_with_ret.sh  openbookqa-main,super_glue-wic  BART 42 5 rerank sentiment
sbatch scripts/ret/zs_with_ret.sh  super_glue-wsc.fixed,winogrande-winogrande_xl  BART 42 5 rerank sentiment
sbatch scripts/ret/zs_with_ret.sh  anli_r3,piqa BART 42 5 rerank sentiment

sbatch scripts/ret/zs_with_ret.sh  ai2_arc-ARC-Challenge,super_glue-cb  BART 42 5 rerank closed_book_qa
sbatch scripts/ret/zs_with_ret.sh  squad_v2,hellaswag BART 42 5 rerank closed_book_qa
sbatch scripts/ret/zs_with_ret.sh  openbookqa-main,super_glue-wic  BART 42 5 rerank closed_book_qa
sbatch scripts/ret/zs_with_ret.sh  super_glue-wsc.fixed,winogrande-winogrande_xl  BART 42 5 rerank closed_book_qa
sbatch scripts/ret/zs_with_ret.sh  anli_r3,piqa BART 42 5 rerank closed_book_qa

sbatch scripts/ret/zs_with_ret.sh  ai2_arc-ARC-Challenge,super_glue-cb  BART 42 5 rerank structure_to_text
sbatch scripts/ret/zs_with_ret.sh  squad_v2,hellaswag BART 42 5 rerank structure_to_text
sbatch scripts/ret/zs_with_ret.sh  openbookqa-main,super_glue-wic  BART 42 5 rerank structure_to_text
sbatch scripts/ret/zs_with_ret.sh  super_glue-wsc.fixed,winogrande-winogrande_xl  BART 42 5 rerank structure_to_text
sbatch scripts/ret/zs_with_ret.sh  anli_r3,piqa BART 42 5 rerank structure_to_text

sbatch scripts/ret/zs_with_ret.sh  ai2_arc-ARC-Challenge,super_glue-cb  BART 42 5 rerank topic_classification
sbatch scripts/ret/zs_with_ret.sh  squad_v2,hellaswag BART 42 5 rerank topic_classification
sbatch scripts/ret/zs_with_ret.sh  openbookqa-main,super_glue-wic  BART 42 5 rerank topic_classification
sbatch scripts/ret/zs_with_ret.sh  super_glue-wsc.fixed,winogrande-winogrande_xl  BART 42 5 rerank topic_classification
sbatch scripts/ret/zs_with_ret.sh  anli_r3,piqa BART 42 5 rerank topic_classification

sbatch scripts/ret/zs_with_ret.sh  ai2_arc-ARC-Challenge,super_glue-cb  BART 42 5 rerank paraphrase_identification
sbatch scripts/ret/zs_with_ret.sh  squad_v2,hellaswag BART 42 5 rerank paraphrase_identification
sbatch scripts/ret/zs_with_ret.sh  openbookqa-main,super_glue-wic  BART 42 5 rerank paraphrase_identification
sbatch scripts/ret/zs_with_ret.sh  super_glue-wsc.fixed,winogrande-winogrande_xl  BART 42 5 rerank paraphrase_identification
sbatch scripts/ret/zs_with_ret.sh  anli_r3,piqa BART 42 5 rerank paraphrase_identification










BIG-Bench tasks  
sbatch scripts/ret/zs_with_ret.sh hindu_knowledge,known_unknowns BART 42 5 rerank
sbatch scripts/ret/zs_with_ret.sh logic_grid_puzzle,movie_dialog_same_or_different BART 42 5 rerank 
sbatch scripts/ret/zs_with_ret.sh strategyqa,vitaminc_fact_verification BART 42 5 rerank 


sbatch scripts/ret/zs_with_ret.sh hindu_knowledge,known_unknowns BART 42 5 no
sbatch scripts/ret/zs_with_ret.sh logic_grid_puzzle,movie_dialog_same_or_different BART 42 5 no 
sbatch scripts/ret/zs_with_ret.sh strategyqa,vitaminc_fact_verification BART 42 5 no 

sbatch scripts/ret/zs_with_ret.sh hindu_knowledge,known_unknowns SentenceTransformer 42 5 no
sbatch scripts/ret/zs_with_ret.sh logic_grid_puzzle,movie_dialog_same_or_different SentenceTransformer 42 5 no 
sbatch scripts/ret/zs_with_ret.sh strategyqa,vitaminc_fact_verification SentenceTransformer 42 5 no 

sbatch scripts/ret/zs_with_ret.sh hindu_knowledge,known_unknowns Random 42 5 no
sbatch scripts/ret/zs_with_ret.sh logic_grid_puzzle,movie_dialog_same_or_different Random 42 5 no 
sbatch scripts/ret/zs_with_ret.sh strategyqa,vitaminc_fact_verification Random 42 5 no 





# sbatch scripts/ret/zs_with_ret.sh  anli_r1,anli_r2  BART 42 5 rerank




sbatch scripts/ret/zs_with_ret.sh  ai2_arc-ARC-Challenge,super_glue-cb  TwoStage-BART-TRAINED 42 5 no
sbatch scripts/ret/zs_with_ret.sh  squad_v2,hellaswag TwoStage-BART-TRAINED 42 5 no
sbatch scripts/ret/zs_with_ret.sh  openbookqa-main,super_glue-wic  TwoStage-BART-TRAINED 42 5 no
sbatch scripts/ret/zs_with_ret.sh  super_glue-wsc.fixed,winogrande-winogrande_xl  TwoStage-BART-TRAINED 42 5 no
sbatch scripts/ret/zs_with_ret.sh  anli_r1,anli_r2  TwoStage-BART-TRAINED 42 5 no
sbatch scripts/ret/zs_with_ret.sh  anli_r3,piqa TwoStage-BART-TRAINED 42 5 no
 



# ---------------------------------------------------------------------------------- # K=10 
sbatch scripts/ret/zs_with_ret.sh  ai2_arc-ARC-Challenge,super_glue-cb  Random 2022 10 no
sbatch scripts/ret/zs_with_ret.sh  squad_v2,hellaswag Random 2022 10 no
sbatch scripts/ret/zs_with_ret.sh  openbookqa-main,super_glue-wic  Random 2022 10 no
sbatch scripts/ret/zs_with_ret.sh  super_glue-wsc.fixed,winogrande-winogrande_xl  Random 2022 10 no
sbatch scripts/ret/zs_with_ret.sh  anli_r1,anli_r2  Random 2022 10 no
sbatch scripts/ret/zs_with_ret.sh  anli_r3,piqa Random 2022 10 no

sbatch scripts/ret/zs_with_ret.sh  ai2_arc-ARC-Challenge,super_glue-cb  Random 2022 10 rerank
sbatch scripts/ret/zs_with_ret.sh  squad_v2,hellaswag Random 2022 10 rerank
sbatch scripts/ret/zs_with_ret.sh  openbookqa-main,super_glue-wic  Random 2022 10 rerank
sbatch scripts/ret/zs_with_ret.sh  super_glue-wsc.fixed,winogrande-winogrande_xl  Random 2022 10 rerank
sbatch scripts/ret/zs_with_ret.sh  anli_r1,anli_r2  Random 2022 10 rerank
sbatch scripts/ret/zs_with_ret.sh  anli_r3,piqa Random 2022 10 rerank


sbatch scripts/ret/zs_with_ret.sh  ai2_arc-ARC-Challenge,super_glue-cb  SentenceTransformer 2022 10 no
sbatch scripts/ret/zs_with_ret.sh  squad_v2,hellaswag SentenceTransformer 2022 10 no
sbatch scripts/ret/zs_with_ret.sh  openbookqa-main,super_glue-wic  SentenceTransformer 2022 10 no
sbatch scripts/ret/zs_with_ret.sh  super_glue-wsc.fixed,winogrande-winogrande_xl  SentenceTransformer 2022 10 no
sbatch scripts/ret/zs_with_ret.sh  anli_r1,anli_r2  SentenceTransformer 2022 10 no
sbatch scripts/ret/zs_with_ret.sh  anli_r3,piqa SentenceTransformer 2022 10 no


sbatch scripts/ret/zs_with_ret.sh  ai2_arc-ARC-Challenge,super_glue-cb  BART 42 10 no
sbatch scripts/ret/zs_with_ret.sh  squad_v2,hellaswag BART 42 10 no
sbatch scripts/ret/zs_with_ret.sh  openbookqa-main,super_glue-wic  BART 42 10 no
sbatch scripts/ret/zs_with_ret.sh  super_glue-wsc.fixed,winogrande-winogrande_xl  BART 42 10 no
sbatch scripts/ret/zs_with_ret.sh  anli_r1,anli_r2  BART 42 10 no
sbatch scripts/ret/zs_with_ret.sh  anli_r3,piqa BART 42 10 no

# sbatch scripts/ret/zs_with_ret.sh  ai2_arc-ARC-Challenge,super_glue-cb  vBART 2022 10 no
# sbatch scripts/ret/zs_with_ret.sh  squad_v2,hellaswag vBART 2022 10 no
# sbatch scripts/ret/zs_with_ret.sh  openbookqa-main,super_glue-wic  vBART 2022 10 no
# sbatch scripts/ret/zs_with_ret.sh  super_glue-wsc.fixed,winogrande-winogrande_xl  vBART 2022 10 no
# sbatch scripts/ret/zs_with_ret.sh  anli_r1,anli_r2  vBART 2022 10 no
# sbatch scripts/ret/zs_with_ret.sh  anli_r3,piqa vBART 2022 10 no
 









sbatch scripts/ret/zs_with_ret.sh  ai2_arc-ARC-Challenge,super_glue-cb  SentenceTransformer 2022 10 rerank
sbatch scripts/ret/zs_with_ret.sh  squad_v2,hellaswag SentenceTransformer 2022 10 rerank
sbatch scripts/ret/zs_with_ret.sh  openbookqa-main,super_glue-wic  SentenceTransformer 2022 10 rerank
sbatch scripts/ret/zs_with_ret.sh  super_glue-wsc.fixed,winogrande-winogrande_xl  SentenceTransformer 2022 10 rerank
sbatch scripts/ret/zs_with_ret.sh  anli_r1,anli_r2  SentenceTransformer 2022 10 rerank
sbatch scripts/ret/zs_with_ret.sh  anli_r3,piqa SentenceTransformer 2022 10 rerank


sbatch scripts/ret/zs_with_ret.sh  ai2_arc-ARC-Challenge,super_glue-cb  BART 2022 10 rerank
sbatch scripts/ret/zs_with_ret.sh  squad_v2,hellaswag BART 2022 10 rerank
sbatch scripts/ret/zs_with_ret.sh  openbookqa-main,super_glue-wic  BART 2022 10 rerank
sbatch scripts/ret/zs_with_ret.sh  super_glue-wsc.fixed,winogrande-winogrande_xl  BART 2022 10 rerank
sbatch scripts/ret/zs_with_ret.sh  anli_r1,anli_r2  BART 2022 10 rerank
sbatch scripts/ret/zs_with_ret.sh  anli_r3,piqa BART 2022 10 rerank


# ---------------------------------------------------------------------------------- #

# CUDA_VISIBLE_DEVICES=1 scripts/ret/zs_with_ret.sh  hellaswag BART 42 5 rerank


sbatch scripts/ret/zs_with_ret.sh  ai2_arc-ARC-Challenge,super_glue-cb  TRAINED 42 5 no
sbatch scripts/ret/zs_with_ret.sh  squad_v2,hellaswag TRAINED 42 5 no
sbatch scripts/ret/zs_with_ret.sh  openbookqa-main,super_glue-wic  TRAINED 42 5 no
sbatch scripts/ret/zs_with_ret.sh  super_glue-wsc.fixed,winogrande-winogrande_xl  TRAINED 42 5 no
sbatch scripts/ret/zs_with_ret.sh  anli_r1,anli_r2  TRAINED 42 5 no
sbatch scripts/ret/zs_with_ret.sh  anli_r3,piqa TRAINED 42 5 no


-----

sbatch scripts/ret/zs_with_ret.sh  ai2_arc-ARC-Challenge BART 42 1 no

--------------------------------------------------
# data_names=($(cat scripts/remaining.tasks))

data_names=($(cat scripts/full_downstream.tasks))
for data_name in "${data_names[@]}"
do
    sbatch scripts/ret/zs_with_ret.sh $data_name SentenceTransformer 42 10 rerank
done


sbatch scripts/ret/zs_with_ret.sh super_glue-multirc SentenceTransformer 42 10



'
