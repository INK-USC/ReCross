# ReCross: Unsupervised Cross-Task Generalization via Retrieval Augmentation


### **_Quick links:_**  [**[Paper]**](https://arxiv.org/abs/2204.07937)   [**[Docs]**](https://inklab.usc.edu/ReCross/) [**[BART0]**](https://huggingface.co/yuchenlin/BART0)

---
This is the repository of the paper, [_**Unsupervised Cross-Task Generalization via Retrieval Augmentation**_](https://arxiv.org/abs/2204.07937), by [_Bill Yuchen Lin_](https://yuchenlin.xyz/), Kangmin Tan, Chris Miller, Beiwen Tian, and [Xiang Ren](http://www-bcf.usc.edu/~xiangren/).
---

## Abstract 
Humans can perform unseen tasks by recalling relevant skills that are acquired previously and then generalizing them to the target tasks, even if there is no supervision at all. In this paper, we aim to improve such cross-task generalization ability of massive multi-task language models such as T0 (Sanh et al., 2021) in an unsupervised setting. We propose a retrieval-augmentation method named ReCross that takes a few unlabelled examples as queries to retrieve a small subset of upstream data and uses them to update the multi-task model for better generalization. Our empirical results show that the proposed ReCross consistently outperforms non-retrieval baselines by a significant margin.


## Installation

```bash
# Create a new conda environment (optional)
conda create -n rex python=3.7 # requires python 3.7
conda activate rex
pip install sentence_transformers # to make a version that can work with torch= 1.8.1  
conda install pytorch==1.8.1 cudatoolkit=10.1.168 -c pytorch
conda install mkl=2019.3=199
pip install datasets py7zr wget jinja2
pip install -e git+https://github.com/bigscience-workshop/promptsource.git#egg=promptsource -U  # for processing the data only. can skip.
pip install ipykernel
pip install higher==0.2.1 scikit-learn==0.24.1 scipy==1.4.1  # Important  
pip install transformers==4.12.5 # Important
# pip install faiss-cpu --no-cache
conda install faiss-cpu -c pytorch
pip install wandb   # wandb login --relogin
pip install rouge nltk
pip install sentencepiece
# pip install simpletransformers
pip install -e git+https://github.com/JonnodT/simpletransformers.git#egg=simpletransformers
pip install git+https://github.com/google/BIG-bench.git # This may take a few minutes
pip install -e ./   # install the source code as `metax` to local

```


## The ReCross datasets 

We follow the split of the T0 paper and use the PromptSource templates to convert all examples to text-to-text formats.

### Upstream data for learning the base models

We combine the examples from nearly 40 upstream tasks into a single processed data file, which can be downloaded here: [https://drive.google.com/file/d/1WB6DM7llv5M-UGgdz2X9xtrCZSy5VgVy/view?usp=sharing](https://drive.google.com/file/d/1WB6DM7llv5M-UGgdz2X9xtrCZSy5VgVy/view?usp=sharing).

We will add docs on how to use the scripts under `metax/prompts/` to generate such upstream data later.

### Test data for evaluating cross-task generalization 

Please check `data/task_name/fewshot.json` for the query examples (there are multiple different sets for each size of the query set: 16, 32, 64, etc.). 

And we will use `data/task_name/test-1000.json` for evaluating the performance on generalizing to each target unseen task.

### The upstream/target split of tasks

We use the exactly the same set of upstream tasks as the T0(-3B) models used for training our BART0 models, while the specific selections of the instances and templates may be different.
We use a similar set of target tasks for evaluating the cross-task generation performance, 10 tasks form the PromptSource and 5 from the BIG-bench. Please check our paper and the appendix for more details.

## BART0: a parameter-efficient alternative of T0-3B

We use the above upstream data to fine-tune the `facebook/bart-large` and obtain the `yuchenlin/BART0` model checkpoint.
Please click this link to access the model checkpoint via HuggingFace: [https://huggingface.co/yuchenlin/BART0](https://huggingface.co/yuchenlin/BART0).
Please refer to the `scripts/upstream_training` folder for more implementation details and hyper-parameters.

## Cross-Task Generalization

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

- ***Build the index in parallel *** （SBERT）

Please run `scripts/zs_retrieval/zeroshot_build_index.sh`.

```bash
# submit multiple indexing jobs in parallel 
for shard_id in {0..7}; # in 8 batches
do
    sbatch scripts/zs_retrieval/zeroshot_build_index.sh Semantic 8 $shard_id
done

# when the above jobs are finished, combine the produced files 
python scripts/zs_retrieval/merge_memory_index.py memory_cache/sbert_memory.pkl 8
```

- ***Build the index in parallel *** （BART0）

```bash 
# submit multiple indexing jobs in parallel 
for shard_id in {0..19};    # in 20 batches
do
    sbatch scripts/zs_retrieval/zeroshot_build_index.sh BART 20 $shard_id
done

# when the above jobs are finished, combine the produced files 
python scripts/zs_retrieval/merge_memory_index.py memory_cache/bart0_memory.pkl 20
```


- ***Run an experiment***

Please check the script `scripts/ret/zs_with_ret.sh` and the `metax/cli.py` to know more configurations. 
Here, one can run a particular experiment that uses a particular retrieval method for a certain target task. 


```bash 
sbatch scripts/ret/zs_with_ret.sh  [task names]  [retriever] 42 5 [no|rerank]
```

The first slot should be a string that is a comma-separated list of target task names. For example, `ai2_arc-ARC-Challenge,super_glue-cb`. Then, this script will run 2 individual process at the same time ***on a single gpu*** for each target task. You can also just input a single task name for the 1st slot to run a single target task at a time --- this depends the memory size of the GPUs you have and the consideration on the trade-off between single-gpu efficiency and the number of required GPUs.

The second slot is the name of the retriever where we have three options here: Random, SentenceTransformer, BART. The index path for each option is listed in the script, which can be customized by yourself.


The 42 and 5 here are the initial random seed and the number of rounds. That is, if we input 5, then the script will run 5 times of retrieval and fine-tuning where each round has a different set of query examples.


The final slot is to indicate whether we want to enable re-ranking stage for the ReCross method, where we will introduce in the next section. Simply put, `no` means we won't enable the reranker for refining the retrieved results form BART0 index. And the configurations about the reranker, e.g., the upsampling ratio and the path to the reranker checkpoint is also listed in the script, which can be customized.


### Train the Re-ranking module for ReCross 

TBA.

The scripts related to the reranker are located in `scripts/distant_supervision` and `metax/distant_supervision`

## Analysis and Visualization 

The above scripts will result in logs, prediction results, and retrieved data (if any). The paths of the saved files can be found in the script, and you can customize the paths if you'd like.
To analyze the performance and the behavior of the results on multiple task, please refer to the `scripts/visualize_scores.py`. We also use `scripts/task_corr.ipynb` to draw heatmaps for understanding the task correlation.



## Contact

The codebase and the documentation are still under development, please stay tuned.

Please email yuchen.lin@usc.edu if you have any questions. 

If you'd like cite us, please use this:

```bibtex
@article{Lin2022UnsupervisedCG,
  title={Unsupervised Cross-Task Generalization via Retrieval Augmentation},
  author={Bill Yuchen Lin and Kangmin Tan and Chris Miller and Beiwen Tian and Xiang Ren},
  journal={ArXiv},
  year={2022},
  volume={abs/2204.07937}
}
```