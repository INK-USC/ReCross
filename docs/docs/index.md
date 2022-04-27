---
layout: default
title: Intro
nav_order: 1
description: "Unsupervised Cross-Task Generalization via Retrieval Augmentation"
permalink: /
last_modified_date: April 5th 2022
---
 
# ReCross | Unsupervised Cross-Task Generalization via Retrieval Augmentation
{: .fs-6 .fw-700 .text-blue-300 }

---
<span class="fs-2">
[Paper](https://arxiv.org/abs/2204.07937){: target="_blank" .btn .btn-green .mr-1 .fs-3}
[Github](https://github.com/INK-USC/ReCross/){: target="_blank" .btn .btn-purple .mr-1 .fs-3 }
[Video](#){: target="_blank" .btn .btn-blue .mr-1 .fs-3 }
[Slides](#){: target="_blank" .btn .btn-red .mr-1 .fs-3 }
</span>


<!--
--- 
<span class="fs-2">
[Data](/data){: .btn .btn-green .mr-1 }
[Methods](/methods){: .btn .btn-purple .mr-1 }
[Metrics](/metrics){: .btn .btn-blue .mr-1 }
[Leaderboard](/leaderboard){: .btn .btn-red .mr-1 }
</span>
-->

---


<!-- ![DrFact](/images/poaster.png){: style="text-align:center; display:block; margin-left: auto; margin-right: auto;" width="100%"} -->

This is the project site for the paper, [_**Unsupervised Cross-Task Generalization via Retrieval Augmentation**_](https://arxiv.org/abs/2204.07937), by [_Bill Yuchen Lin_](https://yuchenlin.xyz/), Kangmin Tan, Chris Miller, Beiwen Tian, and [Xiang Ren](http://www-bcf.usc.edu/~xiangren/).


 
---

<!-- 
 <style type="text/css">
    .image-left {
      display: block;
      margin-left: auto;
      margin-right: auto;
      float: right;
    }
 
    table th:first-of-type {
        width: 10
    }
    table th:nth-of-type(2) {
        width: 10
    }
    table th:nth-of-type(3) {
        width: 50
    }
    table th:nth-of-type(4) {
        width: 30
    } 

    </style> -->





 
<!-- {: .fs-3 .fw-300 } -->
## Abstract
Humans can perform unseen tasks by recalling relevant skills that are acquired previously and then generalizing them to the target tasks, even if there is no supervision at all. In this paper, we aim to improve such cross-task generalization ability of massive multi-task language models such as T0 (Sanh et al., 2021) in an unsupervised setting. We propose a retrieval-augmentation method named ReCross that takes a few unlabelled examples as queries to retrieve a small subset of upstream data and uses them to update the multi-task model for better generalization. Our empirical results show that the proposed ReCross consistently outperforms non-retrieval baselines by a significant margin.

---
## Problem Formulation

![Introduction of the problem](images/rex_intro.gif){: style="text-align:center; display:block; margin-left: auto; margin-right: auto; border: 2px solid black;" width="95%"}


---

## ReCross: the upstream stage

![Introduction of the ReCross](images/rex_method_train.gif){: style="text-align:center; display:block; margin-left: auto; margin-right: auto; border: 2px solid black;" width="95%"}

--- 

## ReCross: the generalization stage

![Introduction of the ReCross](images/rex_method_test.gif){: style="text-align:center; display:block; margin-left: auto; margin-right: auto; border: 2px solid black;" width="95%"}

---

## Cite

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