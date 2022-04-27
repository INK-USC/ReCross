#!/bin/bash
#SBATCH --time 1:00:00
#SBATCH --gres gpu:1 
#SBATCH --output jobs/%j.out
#SBATCH --job-name zeroshot-test 

RETRIEVE_SEED=$2
TARGET_TASK=$1
MODEL="BART0"
OUTPUT_DIR="results/$MODEL-zeroshot-retriever/random-$RETRIEVE_SEED-$TARGET_TASK"
LOG_DIR="logs/$MODEL-zeroshot-retriever/random-$RETRIEVE_SEED-$TARGET_TASK"
MODEL_NAME="BART0-Random-$2-3331-3e-6-200"    # checkpoints/BART0-Random-1337-3331-5e-6-100.pt
MODEL_PATH="checkpoints/$MODEL_NAME.pt"
mkdir -p $OUTPUT_DIR
mkdir -p $LOG_DIR

python -m metax.run \
    --action="zero_shot_evaluation" \
    --run_name="${MODEL_NAME}-zs-${TARGET_TASK}" \
    --model_type="yuchenlin/$MODEL" \
    --log_dir=$LOG_DIR \
    --checkpoint="checkpoints/" \
    --output_dir=$OUTPUT_DIR \
    --model_path=$MODEL_PATH \
    --predict_batch_size 8 \
    --prefix "-" \
    --target_task="${TARGET_TASK}" \
    --max_input_length 512 

# To run this script:
# sbatch scripts/zs_retrieval/zeroshot_random_test_one.sh super_glue-rte 42
