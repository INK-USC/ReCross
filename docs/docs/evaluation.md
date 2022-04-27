---
layout: default
title: Evaluation
nav_order: 4
# toc_list: true
last_modified_date: April 5th 2022
permalink: /evaluation
mathjax: true
has_toc: true
---

# Evaluation & Analysis
{: .no_toc}




## Table of contents
{: .no_toc .text-delta }

- TOC
{:toc}


## Analysis and Visualization 

The above scripts will result in logs, prediction results, and retrieved data (if any). The paths of the saved files can be found in the script, and you can customize the paths if you'd like.
To analyze the performance and the behavior of the results on multiple task, please refer to the `scripts/visualize_scores.py`. We also use `scripts/task_corr.ipynb` to draw heatmaps for understanding the task correlation.

## Main Experimental Results 

![Introduction of the ReCross](images/main_res.png){: style="text-align:center; display:block; margin-left: auto; margin-right: auto; border: 2px solid black;" width="95%"}


## Analysis on Task Correlation

![Introduction of the ReCross](images/rex_corr.png){: style="text-align:center; display:block; margin-left: auto; margin-right: auto; border: 2px solid black;" width="95%"}