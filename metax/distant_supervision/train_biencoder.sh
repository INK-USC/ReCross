#!/bin/bash
#SBATCH --time=12:00:00
#SBATCH --job-name=metax_biencoder_train
#SBATCH --output jobs/%j.out
#SBATCH --gres=gpu:6000:1
#sSBATCH --qos=general

python -m metax.distant_supervision.train_biencoder \
  --max_input_length 512 \
  --max_output_length 64 \
  --learning_rate 3e-3 \
  --train_batch_size 8 \
  --n_epochs 16 \
  --model_type 'yuchenlin/BART0' \
  --ret_merge_mode 'unsupervised' \
  --biencoder_model_path "checkpoints/biencoder_zs/"

# sbatch metax/distant_supervision/train_biencoder.sh