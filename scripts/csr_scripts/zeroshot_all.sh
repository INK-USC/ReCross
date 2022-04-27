#!/bin/bash
data_names=($(cat scripts/csr_downstream.tasks))
# declare -a models=("T0_3B")
# declare -a models=("BART0")
# declare -a models=("BART0" "BART0pp")
# declare -a models=("BART0pp-base" "BART0-base")
declare -a models=("BART0_CSR")
for data_name in "${data_names[@]}" 
do
for model_name in "${models[@]}" 
do
    sbatch scripts/no_ret/zeroshot_one.sh ${data_name} ${model_name}
    echo "Submitted: ${data_name} with ${model_name}"
done
done