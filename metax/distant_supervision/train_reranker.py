from simpletransformers.classification import ClassificationModel, ClassificationArgs
import json
import pandas as pd
import logging
import sklearn
from tqdm import tqdm
import random
import argparse
import os
import torch

from itertools import product

os.environ['TOKENIZERS_PARALLELISM'] = 'False'

DEFAULT_RUNNAME = random.randrange(1,100)

# Paths
DS_TRAIN_DATA_PATH = "data/ds_from_bart0_upstream_train.json"
DS_EVAL_DATA_PATH = "data/ds_from_bart0_upstream_eval.json"
DS_MODEL_PATH = "checkpoints/reranker_roberta_base/"
PARTITIONED_DATA_PATH= "data/ds_copy/"


main_logger = logging.getLogger("data_preprocess")
main_logger.setLevel(logging.INFO)


def make_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def gen_one(sa, sb, label):
    """
        Generate one reranker training data instance
        
        Parameters:
        ----------
            sa: task instance a
            sb: task instance b
            label: 0 or 1, whether the two are of the same type. 
    """
    return [sa[0], sb[0], label]

def save_copy_of_data(train_data, dev_data, test_data, args):
    train_data_path = args.partitioned_data_path + "train.json"
    dev_data_path = args.partitioned_data_path + "dev.json"
    test_data_path = args.partitioned_data_path + "test.json"
    json.dump(train_data, open(train_data_path, 'w'), indent=4)
    json.dump(dev_data, open(dev_data_path, 'w'), indent=4)
    json.dump(test_data, open(test_data_path, 'w'), indent=4)

def training_create_paths(args):
    args.reranker_model_path += args.runname + "/"
    make_dir(args.reranker_model_path)
    args.partitioned_data_path += args.runname + "/"
    make_dir(args.partitioned_data_path)

def inference_create_paths(args):
    args.output_data_path += args.runname + "/"
    make_dir(args.output_data_path)


def assemble_pairs(rdata, cutoff=-1, mode="by_row"):
    all_data = []
    if cutoff > 0:
        rdata = random.sample(rdata, cutoff)
    print(f"len(rdata)={len(rdata)}")


    if mode == "by_row":
        # 2 pairs for each row, 1 positive, 1 negative
        for tp in tqdm(rdata):
            # Iterate each 3-tuple
            que = tp['query_examples']
            pos = tp['positive_examples']
            neg = tp['negative_examples']
            sz = min(len(que), len(pos), len(neg))
            for i in range(sz):
                # Create 1 positive entry and 1 negative entry
                all_data.append(gen_one(que[i],pos[i],1))
                all_data.append(gen_one(que[i],neg[i],0))
    elif mode == "product":
        for tp in tqdm(rdata):
            # Iterate each 3-tuple
            que = tp['query_examples']
            pos = tp['positive_examples']
            neg = tp['negative_examples']

            # Use cartesian product
            positive_pairs = [gen_one(q, p, 1) for q,p in product(que,pos)]
            negative_pairs = [gen_one(q, n, 0) for q,n in product(que,neg)]

            all_data += positive_pairs
            all_data += negative_pairs
    return all_data

def create_model_dataframe(data):
    """
        Convert data to the dataformat required by
        classification model
    """

    model_df = pd.DataFrame(data)
    model_df.columns = ["text_a", "text_b", "labels"]

    return model_df

def prep_data(args):
    """
        Load raw ds data and prepare pair dataset for training
    """
    train_rdata = json.load(open(args.ds_train_data_path))
    dev_data = json.load(open(args.ds_dev_data_path))
    test_data = json.load(open(args.ds_test_data_path))

    train_data = assemble_pairs(train_rdata, args.ds_train_cut_off, mode="product")
    dev_data = assemble_pairs(dev_data, args.ds_eval_cut_off, mode="product")
    test_data = assemble_pairs(test_data, args.ds_eval_cut_off, mode="product")
    

    random.seed(42)
    random.shuffle(train_data)
    # random.shuffle(eval_data)

    save_copy_of_data(train_data, dev_data, test_data, args)

    train_df = create_model_dataframe(train_data)
    dev_df = create_model_dataframe(dev_data)
    test_df = create_model_dataframe(test_data)

    main_logger.info(f"Training Dataset Size {len(train_data)} | Evalutation Dataset Size {len(dev_data)} | Evalutation Dataset Size {len(test_data)}")

    return train_df, dev_df, test_df


def get_parser():
    parser = argparse.ArgumentParser(description="Traine Reranker")
    parser.add_argument("--mode", type=str, default="train", help="Select from ['train','infer']")
    parser.add_argument("--runname", type=str, default=f"Run_{DEFAULT_RUNNAME}", help="Name of run")
    parser.add_argument('--reranker_model_path', type=str, default=DS_MODEL_PATH, help="Path to store the checkpoints.")

    # Parameters for training
    parser.add_argument('--ds_train_data_path', type=str, default=DS_TRAIN_DATA_PATH, help="Path to distant supervision data (Training).")
    parser.add_argument('--ds_dev_data_path', type=str, default=DS_EVAL_DATA_PATH, help="Path to distant supervision data (dev).")
    parser.add_argument('--ds_test_data_path', type=str, default=DS_EVAL_DATA_PATH, help="Path to distant supervision data (test).")
    parser.add_argument('--partitioned_data_path', type=str, default=PARTITIONED_DATA_PATH, help="Where to save a copy of the partitioned train/eval data.")
    parser.add_argument('--ds_train_cut_off', type=int, default=-1, help="sample the training data, if positive")
    parser.add_argument('--ds_eval_cut_off', type=int, default=-1, help="sample the training data, if positive")
    
    # Parameters for inference
    parser.add_argument('--input_data_path', type=str, help="Path to instances to classify")
    parser.add_argument('--output_data_path', type=str, help="Where to store the output of the model")

    return parser

def main():
    args = get_parser().parse_args()
    logging.basicConfig(level=logging.INFO)
    transformers_logger = logging.getLogger("transformers")
    transformers_logger.setLevel(logging.WARNING) 
    main_logger.info(f"--------------------------{args.runname}--------------------------")

    if args.mode == "train":
        training_create_paths(args)
        main_logger.info(f"Mode: {args.mode}")


        train_df, dev_df, test_df = prep_data(args)
        # Optional model configuration
        n_gpu = torch.cuda.device_count()
        model_args = ClassificationArgs(num_train_epochs=5, 
                                        # tokenizer_type="roberta",
                                        # tokenizer_name="roberta-base",
                                        use_hf_datasets=True,
                                        no_cache=False,
                                        do_lower_case=True,
                                        fp16=True,
                                        overwrite_output_dir=True,
                                        train_batch_size=64, 
                                        eval_batch_size=64, 
                                        output_dir=args.reranker_model_path,
                                        max_seq_length=512,
                                        learning_rate=1e-5,
                                        warmup_steps=100,
                                        n_gpu=n_gpu,
                                        evaluate_during_training=True,
                                        evaluate_during_training_verbose=True,
                                        evaluate_during_training_steps=100,
                                        save_steps=100,
                                        wandb_project=args.runname)
        print(model_args)
        # Create a ClassificationModel
        model = ClassificationModel(
            "bart", "facebook/bart-base", args=model_args, num_labels=2
        )

        # Train the model
        model.train_model(train_df, eval_df=dev_df)

        # Evaluate the model
        result, model_outputs, wrong_predictions = model.eval_model(dev_df, f1=sklearn.metrics.f1_score, acc=sklearn.metrics.accuracy_score)
        print(result)
        result, model_outputs, wrong_predictions = model.eval_model(test_df, f1=sklearn.metrics.f1_score, acc=sklearn.metrics.accuracy_score)
        print(result)

    elif args.mode == "infer":
        inference_create_paths(args)
        main_logger.info(f"Mode: {args.mode}")
        main_logger.info(f"Loading Model from {args.reranker_model_path}")
        # Create a ClassificationModel
        model = ClassificationModel(
            "roberta", args.reranker_model_path
        )
        # Load input data
        # data = json.load(open(args.input_data_path)) # A list of sentence pairs

        data = [
            ("This is an input sentence.", "This is an output sentence."),
            ("This is an output sentence.", "This is an input sentence."),
            ("This is an input sentence.", "This is not an output sentence."),
            ("This is an input sentence.", "That is an apple on the tree."),
        ]


        preds, model_output = model.predict(data)

        print(preds)
        print(model_output)

        # Output prediction 1 or 0
        json.dump(list(zip(data,preds)), open(args.output_data_path + "predictions.json",'w'), indent=4)

        # Output the scores output by the model to facilitate ranking.
        json.dump(list(zip(data,list([v[0], v[1]] for v in model_output))), open(args.output_data_path + "scores.json",'w'), indent=4)
    else:
        main_logger.info(f"Unknown mode:{args.mode}. Please choose from ['train', 'infer']")


if __name__ == "__main__":
    print("Warning: this version is deprecated! All the reranker training is done by scripts in metax/reranker_bootstrap")