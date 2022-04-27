## Reranker Bootstrapping Pipeline 
---
* `gen_bart_queries` to select queries (list of ~8 instances). The way it does it is selecting from ds data (not the raw upstream training data).

* `bart_retrieve` uses the queries generated from the previous step and retrieve with bart0 retriever (it filters out examples from the same **task**). It can shard the result into pieces. This is because `gen_better_ds` is very slow so we want to parallelize it on multiple GPUs. 

* For each bootstrapping iteration

    * `gen_better_ds` takes the input from `bart_retrieve` (1 shard per GPU), and generate better distant supervision data. The logic is: for each [query, retrieved] pair: rerank (if not the first iteration) -> take top 100 -> sort by loss delta on test data (data from same task as query) -> top as positive, bottom as negative. 

    * `tune_reranker` if this is first iteration. Train it from scratch, otherwise train it with a smaller learning rate. 