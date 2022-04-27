#!/bin/bash
#SBATCH --time=1:30:00
#SBATCH --job-name=few_shot_only
#SBATCH --output jobs/%j.out

#sSBATCH --gres=gpu:1 
#sSBATCH --qos=general-8000
#sSBATCH --exclude ink-nova

#SBATCH --gres=gpu:6000:1 
#SBATCH --qos=general


EARLY_STOPPING=True
# TARGET_TASK=$1
LR=1e-5
EPOCH=5
MODEL="yuchenlin/BART0_CSR"
MODEL_STR="BART0_CSR"
# MODEL="facebook/bart-large"
# MODEL_STR="BART"


MODE="none" 
RETRIEVE_SEED=1337
SHOTS=64
LOG_DIR="logs/csr-fewshot"
RESULT_DIR="results/csr-fewshot"
CKPT_DIR="checkpoints"
mkdir -p $LOG_DIR
mkdir -p $RESULT_DIR
mkdir -p $CKPT_DIR


TASKS_IN=$1

tasks=$(echo $TASKS_IN | tr "," "\n")

for task in $tasks
do
TARGET_TASK=$task

echo $RETRIEVER
run_name="FS_${MODEL_STR}_${MODE}-${SHOTS}-${EARLY_STOPPING}-${TARGET_TASK}-${LR}-${EPOCH}"
echo ${run_name}


python -m metax.run \
    --run_name "${run_name}" \
    --ret_merge_mode $MODE \
    --action "few_shot_evaluation" \
    --log_dir $LOG_DIR \
    --checkpoint $CKPT_DIR \
    --num_train_epochs_fs ${EPOCH} \
    --num_shots $SHOTS \
    --warmup_steps 0 \
    --learning_rate $LR \
    --train_batch_size 6 \
    --gradient_accumulation_steps 1 \
    --predict_batch_size 4 \
    --target_task $TARGET_TASK \
    --output_dir $RESULT_DIR \
    --early_stopping $EARLY_STOPPING \
    --max_input_length 512 \
    --max_input_length_for_eval 512 \
    --model_type ${MODEL} \
    --retrieve_seed $RETRIEVE_SEED \
    --test_on_the_fly True \
    --finetune_round 5 &
done
wait;

: '

data_names=($(cat scripts/csr_downstream.tasks))
for data_name in "${data_names[@]}" 
do
    sbatch scripts/csr_scripts/few_shot.sh ${data_name}
done
 

# sbatch scripts/csr_scripts/few_shot.sh  ag_news,anli_r1
# sbatch scripts/csr_scripts/few_shot.sh  anli_r2,anli_r3

sbatch scripts/csr_scripts/few_shot.sh  ag_news,rotten_tomatoes
sbatch scripts/csr_scripts/few_shot.sh  glue-mrpc,glue-qqp
sbatch scripts/csr_scripts/few_shot.sh  imdb,squad_v2
sbatch scripts/csr_scripts/few_shot.sh  super_glue-boolq,super_glue-wic
sbatch scripts/csr_scripts/few_shot.sh  super_glue-cb,super_glue-rte

'

 