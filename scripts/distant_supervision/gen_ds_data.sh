python metax/distant_supervision/ds_gen.py \
    --eval_reserve task \
    --N_eval 3000 \
    --N_train 30000 \
    --n_query 8  --n_positive 8 --n_negative 8 
# this will output "data/ds_from_bart0_upstream_[train/dev/test].json"