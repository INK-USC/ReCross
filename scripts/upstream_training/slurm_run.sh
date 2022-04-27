declare -a CKPTs=("86000" "84000" "82000" "80000" "78000" "76000" "74000" "72000" "70000" "68000" "66000" "64000" "62000" "60000" "58000" "56000" "54000" "52000" "50000" "48000" "46000" "44000" "42000" "40000" "38000" "36000" "34000" "32000" "30000" "28000")
for CKPT in "${CKPTs[@]}"
do
session_name=BART0pp_${CKPT}
tmux new-session -d -s ${session_name} "srun --job-name ${session_name} --gpus-per-node=1 --constraint=volta32gb --partition=devlab --time=30 --cpus-per-task 4 --pty scripts/fair_scripts/upstream_validation.sh BART0pp ${CKPT}"
echo "Created tmux session: ${session_name}"
done



declare -a CKPTs=("56000" "54000" "52000" "50000" "48000" "46000" "44000" "40000" "38000" "36000" "34000" "32000" "30000" "28000" "26000" "24000" "22000" "20000" "18000" "16000" "14000" "12000" "10000" "8000" "6000" "4000" "2000")
for CKPT in "${CKPTs[@]}"
do
session_name=BART0_${CKPT}
tmux new-session -d -s ${session_name} "srun --job-name ${session_name} --gpus-per-node=1 --constraint=volta32gb --partition=devlab --time=30 --cpus-per-task 4 --pty scripts/fair_scripts/upstream_validation.sh BART0 ${CKPT}"
echo "Created tmux session: ${session_name}"
done


