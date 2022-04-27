SHARDS=6
for (( i=1; i<=SHARDS; i++ ));
do
    sbatch ./scripts/reranker_bootstrap/gen_better_ds_one.sh $i $SHARDS
done