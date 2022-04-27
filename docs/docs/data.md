---
layout: default
title: Data
nav_order: 2
# toc_list: true
last_modified_date: April 5th 2022
permalink: /data
has_toc: true
---

# The ReCross datasets 
{: .no_toc}







## Table of contents
{: .no_toc .text-delta }

- TOC
{:toc}
 
---

 

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