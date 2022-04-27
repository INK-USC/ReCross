# How to train & use reranker

To train reranker
* First make sure you have generated ds data using `gen_ds_data.sh`
* Run `train_reranker.sh`. There are default values for where the ds train/eval data is located. Make sure to specify path to save the model (`--reranker_model_path`). Note that there's a parameter `--runname`. This argument is used to distinguish different runs. When saving the model and copy of traning data, the program will create folders named after `runname`, so you don't have to specify runname-specific folder names yourself. 


Use `reranker_rank.sh` to do inference. The input data should be a list of string pairs (i.e. N by 2) as json. There would be 2 output files. `predictions.json` will have the 0 or 1 prediction for each sentence pair. `scores.json` will have the raw model output [x,y] for each sentence pair to facilitate ranking. Again, you just need to specify a generic folder as your `--output_data_path` and the program will create a folder named after `--runname` for you to distinguish different runs. 
