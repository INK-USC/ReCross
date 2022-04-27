#!/bin/bash
#SBATCH --time=0:30:00
#SBATCH --job-name=metax_zs_eval
#SBATCH --output jobs/%j.out
#SBATCH --gres=gpu:1
#sSBATCH --exclude ink-ron
#sSBATCH --qos=general

TARGET_TASK=$1
MODEL=$2
# echo $MODEL
if [[ "$MODEL" == "T0"* ]]; then
    # T0-3B, etc.
    model_type="bigscience/$MODEL"
elif [[ "$MODEL" == "BART"* ]]; then
    # BART0pp, BART0, etc.
    model_type="yuchenlin/$MODEL"
fi
echo "MODEL=$MODEL" 
echo "TARGET_TASK=$TARGET_TASK" 

OUTPUT_DIR="results/$MODEL-zeroshot"
LOG_DIR="logs/$MODEL-zeroshot"
mkdir -p $OUTPUT_DIR
mkdir -p $LOG_DIR
 

python -m metax.run \
    --append_another_bos True \
    --action="zero_shot_evaluation" \
    --run_name="$MODEL-zs-${TARGET_TASK}" \
    --model_type=${model_type} \
    --log_dir=$LOG_DIR \
    --checkpoint="checkpoints/" \
    --output_dir=$OUTPUT_DIR \
    --predict_batch_size 4 \
    --prefix "-" \
    --target_task="${TARGET_TASK}" \
    --max_input_length 512
     
# To run this script:
# sbatch scripts/no_ret/zeroshot_one.sh anli_r1 T0_3B
# sbatch scripts/no_ret/zeroshot_one.sh squad_v2 BART0
 
: '
#!/bin/bash

data_names=($(cat scripts/full_downstream.tasks))
declare -a models=("T0_3B")
# declare -a models=("BART0")
# declare -a models=("BART0" "BART0pp")
# declare -a models=("BART0pp-base" "BART0-base")
for data_name in "${data_names[@]}" 
do
for model_name in "${models[@]}" 
do
    sbatch scripts/no_ret/zeroshot_one.sh ${data_name} ${model_name}
    echo "Submitted: ${data_name} with ${model_name}"
done
done


sbatch scripts/csr_scripts/zeroshot_one.sh anli_r1 BART0_CSR
sbatch scripts/csr_scripts/zeroshot_one.sh anli_r2 BART0_CSR
sbatch scripts/csr_scripts/zeroshot_one.sh anli_r3 BART0_CSR
sbatch scripts/csr_scripts/zeroshot_one.sh  super_glue-cb BART0_CSR
sbatch scripts/csr_scripts/zeroshot_one.sh  super_glue-rte BART0_CSR
sbatch scripts/csr_scripts/zeroshot_one.sh  super_glue-wic BART0_CSR
sbatch scripts/csr_scripts/zeroshot_one.sh  super_glue-wsc.fixed BART0_CSR
sbatch scripts/csr_scripts/zeroshot_one.sh  winogrande-winogrande_xl BART0_CSR



'