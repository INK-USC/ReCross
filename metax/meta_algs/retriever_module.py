import pickle
import logging
from tqdm import tqdm
from argparse import ArgumentParser
import json

import numpy as np

from metax.datasets import get_formatted_data
from metax.models.mybart import MyBart
from metax.models.utils import (convert_model_to_single_gpu, freeze_embeds, trim_batch)
from metax.task_manager.dataloader import GeneralDataset
# ImportError: cannot import name '_prepare_bart_decoder_inputs' from 'transformers.models.bart.modeling_bart'
# from transformers.models.bart.modeling_bart import _prepare_bart_decoder_inputs
from transformers import BartTokenizer

import torch
from torch import Tensor
from torch.nn import functional as F
from torch.nn import Module
import random


class BaseIndexManager():
    def __init__(self, logger, args=None):
        super().__init__()
        self.name = "base_index_manager"
        self.logger=logger
        self.index_args = args
        self.memory_examples = {}   # <-- store_memory()

    def get_memory_size(self):
        return len(self.memory_examples)

    def load_memory_from_path(self, init_memory_cache_path):
        raise NotImplementedError

    def save_memory_to_path(self, memory_pkl_path):
        raise NotImplementedError

    def retrieve_from_memory(self, query_examples, sample_size, **kwargs):
        raise NotImplementedError

    def store_examples(self, examples):
        raise NotImplementedError

class BaseMemoryManager:
    def __init__(self):
        pass

class RandomMemoryManger(BaseMemoryManager):
    def __init__(self, logger, args=None):
        # super().__init__(self, logger, args=args)
        self.logger = logger
        self.name = "random_memory_manager"
        self.memory_examples = {}

    # def set_up_initial_memory(self, initial_memory_path="", formatted_examples=None):
    #     self._load_init_memory_examples(initial_memory_path, formatted_examples)

    def load_memory_from_path(self, init_memory_cache_path):
        with open(init_memory_cache_path, "rb") as f:
            memory_cache = pickle.load(f)
            self.logger.info(f"Load the cache to {f.name}")
        self.memory_examples = memory_cache["memory_examples"]

    def save_memory_to_path(self, memory_pkl_path):
        memory_cache = {}
        memory_cache["memory_examples"] = self.memory_examples
        with open(memory_pkl_path, "wb") as f:
            pickle.dump(memory_cache, f)
            self.logger.info(f"Saved the cache to {f.name}")        

    def retrieve_from_memory(self, query_examples=None, sample_size=-1, seed=None, backup_json_path=None, **kwargs):
        assert sample_size > 0
        self.logger.info("Randomly retrieve from the memory. `query_examples` not used")
        key_len = len(self.memory_examples.keys())
        if (sample_size > key_len):
            self.logger.warning(f"Tried to sample {sample_size} items from a population of {key_len}. Sampling all instead.")
            sample_size = key_len
        sample_random = random.Random(seed)  
        self.logger.info(f"seed={seed} in retrieve_from_memory")
        retrieved_example_ids = sample_random.sample(sorted(list(self.memory_examples.keys())), sample_size)
        self.logger.info(f"retrieved_example_ids[:5] = {retrieved_example_ids[:5]}")
        retrieved_examples = [self.memory_examples[rid] for rid in retrieved_example_ids]  
        return retrieved_examples

    def store_examples(self, examples, mode="input"):
        for item in examples:
            # Note that we only keep the all answers here now.
            self.memory_examples[item[2]] = (item[0], item[1], item[2])
        self.logger.info(f"Save {len(self.memory_examples)} examples to the memory.")
