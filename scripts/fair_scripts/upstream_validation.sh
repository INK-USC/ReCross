#!/bin/bash
#SBATCH --time 1:00:00
#SBATCH --gres gpu:6000:1
#SBATCH --job-name zeroshot

TARGET_TASK=all
MODEL=$1
CKPT=$2
OUTPUT_DIR="results/$MODEL-upstream_validation"
mkdir $OUTPUT_DIR
LOG_DIR="logs/$MODEL-upstream_validation"
mkdir -p $OUTPUT_DIR
mkdir -p $LOG_DIR

python -m metax.run \
    --action="zero_shot_evaluation" \
    --run_name="upstream_validation-${MODEL}-${CKPT}" \
    --model_type="./outputs/${MODEL}/checkpoint-${CKPT}" \
    --log_dir=$LOG_DIR \
    --checkpoint="checkpoints/" \
    --output_dir=$OUTPUT_DIR \
    --predict_batch_size 32 \
    --prefix "-" \
    --target_task="${TARGET_TASK}" \
    --max_input_length 512
     
# To run this script:
# scripts/fair_scripts/upstream_validation.sh 84000
 
: '
#!/bin/bash
data_names=($(cat scripts/downstream.tasks))
for data_name in "${data_names[@]}" 
do
    sbatch scripts/zeroshot_bart0.sh ${data_name}
    echo "Submitted: ${data_name}"
done  

grep "Evaluate.*" logs/BART0-large-zeroshot/*.log
'