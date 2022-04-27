---
layout: default
title: Methods
nav_order: 3
has_children: false
has_toc: false
permalink: /methods/
---

# The ReCross & baselines methods
{: .no_toc}

---


## Table of contents
{: .no_toc .text-delta }

- TOC
{:toc}



## BART0: a parameter-efficient alternative of T0-3B

We use the above upstream data to fine-tune the `facebook/bart-large` and obtain the `yuchenlin/BART0` model checkpoint.
Please click this link to access the model checkpoint via HuggingFace: [https://huggingface.co/yuchenlin/BART0](https://huggingface.co/yuchenlin/BART0).
Please refer to the `scripts/upstream_training` folder for more implementation details and hyper-parameters.

## Cross-Task Generalization Methods

### Common pipeline
All experiments will start with `metax/run.py` and the common pipeline is based on `MetaX_Common` and `MetaX_Retrieve` class which are located in `metax/commons.py` and `metax/retrieve.py` respectively. 
The former is the base class for some basic utility functions and support the normal, non-retrieval augmentation methods for cross-task generalization.
The latter is the base class for all retrieval augmentation methods. 

### Running Non-Retrieval Methods 

To run an experiment for such a method, please check `scripts/no_ret/zeroshot_one.sh`, where we have shown the example usages for running multiple target tasks with different base LMs such as BART0 and T0-3B.

Note that this script is a `sbatch` script for submitting a gpu job via the Slurm system, although one can also use it as if it is a standard bash script file.
If you'd like to submit multiple jobs in a batch, please refer to the `scripts/no_ret/zeroshot_all.sh`.

### Running Retrieval-based Methods 

For both SBERT and ReCross methods, we will need to build the dense index of upstream data first before we use them for retrieving additional data based on query examples.

- ***Build the index in parallel*** （SBERT）

Please run `scripts/ret/build_index.sh`.

```bash
# submit multiple indexing jobs in parallel 
for shard_id in {0..15};
do
    sbatch scripts/ret/build_index.sh Semantic 16 $shard_id
done
# when the above jobs are finished, combine the produced files 
python scripts/ret/merge_memory_index.py memory_cache/sbert_memory.pkl 16
```

- ***Build the index in parallel*** （BART0）

```bash 
# submit multiple indexing jobs in parallel 
for shard_id in {0..31};
do
    sbatch scripts/ret/build_index.sh BART0 32 $shard_id
done

# when the above jobs are finished, combine the produced files 
python scripts/ret/merge_memory_index.py memory_cache/bart0_memory.pkl 32
```

## The Reranker of ReCross 

TBA.

The scripts related to the reranker are located in `scripts/distant_supervision` and `scripts/reranker_bootstrap` +  `metax/distant_supervision` and `metax/reranker_bootstrap`.