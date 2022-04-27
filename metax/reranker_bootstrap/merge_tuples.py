"""
    Merge all the json file containing better tuples in one iteration
"""
import argparse
import os
from os import walk
import json

from metax.reranker_bootstrap.utils import *


def get_parser():
    parser = argparse.ArgumentParser(
        description="merge json files")

    parser.add_argument('--data_save_path', type=str,
                        help="Same as data_save_path of gen_one_group")
    
    parser.add_argument('--split_ratio', type=float, default=-1, help="How much to split dev set")

    return parser


def main():
    args = get_parser().parse_args()
    data_save_path = args.data_save_path
    folders = get_folders(data_save_path)  # [] if no file
    print(folders)
    all_data = []
    for fd in folders:
        data_file = get_files(data_save_path + fd)[0]
        fname = data_save_path + fd + "/" + data_file
        data = load_data_from_json(fname)
        all_data += data
    
    
    N = len(all_data)
    print(f"Merged files, a total of {N} tuples.")
    save_json(data_save_path, "all_tuples.json", all_data)
    
    
    if args.split_ratio > 0:
        dev_sz = int(N*args.split_ratio)
        dev_set = all_data[:dev_sz]
        train_set = all_data[dev_sz:]
        
        print(f"Train set size {len(train_set)}")
        save_json(data_save_path, "train_tuples.json", train_set)
        
        print(f"Dev set size {len(dev_set)}")
        save_json(data_save_path, "dev_tuples.json", dev_set)

if __name__ == "__main__":
    main()
