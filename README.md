# ReCross: Unsupervised Cross-Task Generalization via Retrieval Augmentation


### **_Quick links:_**  [**[Paper]**](https://arxiv.org/abs/2204.07937)   [**[Documentation]**](https://inklab.usc.edu//ReCross/)

---
This is the repository of the paper, [_**Unsupervised Cross-Task Generalization via Retrieval Augmentation**_](https://arxiv.org/abs/2204.07937), by [_Bill Yuchen Lin_](https://yuchenlin.xyz/), Kangmin Tan, Chris Miller, Beiwen Tian, and [Xiang Ren](http://www-bcf.usc.edu/~xiangren/).
---

## Abstract 
Humans can perform unseen tasks by recalling relevant skills that are acquired previously and then generalizing them to the target tasks, even if there is no supervision at all. In this paper, we aim to improve such cross-task generalization ability of massive multi-task language models such as T0 (Sanh et al., 2021) in an unsupervised setting. We propose a retrieval-augmentation method named ReCross that takes a few unlabelled examples as queries to retrieve a small subset of upstream data and uses them to update the multi-task model for better generalization. Our empirical results show that the proposed ReCross consistently outperforms non-retrieval baselines by a significant margin.


## Installation

```bash
# Create a new conda environment (optional)
conda create -n metax python=3.7 # requires python 3.7
conda activate metax
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
pip install -e ./   # install the metax to local

```

To run the experiments: go to `scripts/`.

## Data Basics

To download the upstream datasets, click [here](https://drive.google.com/drive/folders/10FSUb3xN_ajmwpwxa7cnjmPTGa8NqmDK?usp=sharing) to download `T0_upstream_train.json` and `T0pp_upstream_train.json` and put it under the `data` folder.
Run below to convert the data format:

```bash
python scripts/convert_upstrem_data_format.py
```
And you will get `data/t0[pp]_upstream_train_lines.json`.


## Zero-Shot Inference + Evaluation

The `scripts/zeroshot_one.sh` is the script for evaluating a base model on a particular target task, and its associated `scripts/zeroshot_all.sh` is used to run multiple interested base models on all tasks.

The results can be summarized by running `scripts/visualize_scores.py`.

## Retrieval-Augmented Zero-Shot Inference + Evaluation

### Random Retrieval

`scripts/retrieval/zeroshot_random_tune_one.sh` is the script for fine-tuning a base model using random retrieval. Note that this script only fine-tunes and saves the model without testing it on any target tasks. And `scripts/retrieval/zeroshot_random_test_one.sh` is used to load a fine-tuned local checkpoint then test it on target tasks. This separation of testing from fine-tuning is an effort to speed up the training process, leveraging random retrieval's independence from target tasks.

`zeroshot_random_tune_all.sh` and `zeroshot_random_test_all.sh` submit multiple fine-tuning/testing jobs at the same time.

### Semantic Retrieval (SentenceBERT-based)
Before running the retrieval, be sure to build the index using `scripts/retrieval/zeroshot_semantic_build_index.sh`. This will build the sentencebert representations of the upstream data and cache them to a file. You can also build the cache implicitly by running the scripts below, however this is not ideal because if you submit more than one job at the same time, both will build the cache (duplicating several hours of work and taking up GPU time). The argument `memory_cache_path` defines the path to the cache. If the upstream data or parameters change (or there is otherwise reason to invalidate the cache) be sure to delete this file; to save compute the script will only rebuild the cache if the file does not exist.

`scripts/retrieval/zeroshot_semantic_one.sh` is used to tune and test a base model using semantic (SentenceBERT-based) retrieval.
`scripts/retrieval/zeroshot_semantic_all.sh` will submit all tune/test jobs at the same time.
