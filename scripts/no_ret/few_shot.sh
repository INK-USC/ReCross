#!/bin/bash
#SBATCH --time=3:00:00
#SBATCH --job-name=few_shot_only
#SBATCH --output jobs/%j.out
#SBATCH --gres=gpu:1 
#SBATCH --qos=general-8000

#sSBATCH --gres=gpu:6000:1 
#sSBATCH --qos=general
#sSBATCH --exclude ink-ron

EARLY_STOPPING=True
TASKS_IN=$1
finetune_round=$2
LR=1e-5
EPOCH=5
MODEL="yuchenlin/BART0"
MODE="none" 
RETRIEVE_SEED=1337
SHOTS=32
LOG_DIR="logs/bart0-fewshot"
RESULT_DIR="results/bart0-fewshot"
CKPT_DIR="checkpoints"
mkdir -p $LOG_DIR
mkdir -p $RESULT_DIR
mkdir -p $CKPT_DIR

echo $RETRIEVER


tasks=$(echo $TASKS_IN | tr "," "\n")

for task in $tasks
do
echo "> [$task]"
TARGET_TASK=$task
run_name="FS_${MODE}-${SHOTS}-${EARLY_STOPPING}-${TARGET_TASK}-${LR}-${EPOCH}"
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
    --train_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --predict_batch_size 4 \
    --target_task $TARGET_TASK \
    --output_dir $RESULT_DIR \
    --early_stopping $EARLY_STOPPING \
    --max_input_length 386 \
    --max_input_length_for_eval 512 \
    --model_type ${MODEL} \
    --retrieve_seed $RETRIEVE_SEED \
    --test_on_the_fly True \
    --finetune_round ${finetune_round} &
done
wait;

: '

data_names=($(cat scripts/full_downstream.tasks))
for data_name in "${data_names[@]}" 
do
    sbatch scripts/no_ret/few_shot.sh ${data_name}
done

# sbatch scripts/no_ret/few_shot.sh ai2_arc-ARC-Easy


sbatch scripts/no_ret/few_shot.sh

sbatch scripts/no_ret/few_shot.sh  ai2_arc-ARC-Easy,ai2_arc-ARC-Challenge 10
sbatch scripts/no_ret/few_shot.sh  squad_v2,hellaswag 10
sbatch scripts/no_ret/few_shot.sh  openbookqa-main,super_glue-multirc  10
sbatch scripts/no_ret/few_shot.sh  super_glue-boolq,super_glue-wic  10
sbatch scripts/no_ret/few_shot.sh  super_glue-wsc.fixed,winogrande-winogrande_xl  10
sbatch scripts/no_ret/few_shot.sh  super_glue-cb,super_glue-rte  10
sbatch scripts/no_ret/few_shot.sh  anli_r1,anli_r2  10
sbatch scripts/no_ret/few_shot.sh  anli_r3,piqa 10

'

 