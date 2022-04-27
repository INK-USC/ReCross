import logging
from argparse import ArgumentParser

from tqdm import tqdm
import numpy as np
import faiss
import math
import random

import torch
from torch import Tensor
from torch.nn import functional as F
from torch.nn import Module
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer

from metax.meta_algs.bartbase_index import BartIndexManager
from metax.datasets import get_formatted_data
from metax.models import run_bart
from metax.models.mybart import MyBart
from metax.models.utils import (convert_model_to_single_gpu, freeze_embeds, trim_batch)
from metax.task_manager.dataloader import GeneralDataset


class SentenceBartIndexManager(BartIndexManager):

    def __init__(self, logger, args=None):
        super().__init__(logger, args=args)
        self.name = "sentence_bart_index_manager"

        self.use_cuda = self.index_args.n_gpu > 0
        self.n_gpu = self.index_args.n_gpu

        self.dim_memory = 768
        self.memory_index = None
        self.memory_examples = {}
        self.memory_example_ids = [] # a sorted list which has the same order of the faiss index.

    def init_model(self):
        self.load_upstream_model(model_type='all-distilroberta-v1')

    def load_upstream_model(self, model_type=None, model_path=None):
        """
        Load model from pretrained or checkpoints
        """
        self.logger.info(f"Loading SentenceTransformer model {model_type} .....")
        model = SentenceTransformer(f'sentence-transformers/{model_type}')

        self.logger.info(f"Loading SentenceTransformer model {model_type} ..... Done!")
        if self.use_cuda:
            if self.n_gpu > 1:
                self.logger.info(f"Setting as dataparallel for {self.n_gpu} gpus.")
                model = torch.nn.DataParallel(model)
            self.logger.info("Moving to the GPUs.")
            model.to(torch.device("cuda"))
        self.upstream_model = model


    def memory_encoding(self, examples, mode="input"):
        # Here we use a simple non-trainable way to store the examples.
        bart_rep_vectors = self.get_representation(examples, mode=mode)
        curr_tensor = torch.tensor(bart_rep_vectors)
        if self.use_cuda:
            curr_tensor = curr_tensor.to('cuda')

        return curr_tensor

    def query_encoding(self, examples, mode="input"):
        # For few-shot examples, we have outputs; for zero-shot examples, we don't have outputs.
        # Here we use a simple non-trainable way to encode the query examples.
        bart_rep_vectors = self.get_representation(examples, mode=mode)
        curr_tensor = torch.tensor(bart_rep_vectors)
        if self.use_cuda:
            curr_tensor = curr_tensor.to('cuda')

        return curr_tensor


    


    def get_representation(self, examples, mode="input"):
        sentence_model = self.upstream_model if (self.n_gpu ==1 or not self.use_cuda) else self.upstream_model.module
        sentence_model.eval()

        if mode.lower() == "both":
            # format to include answer
            # use special token as separator between input/output
            sentences = [f"{example[0]} Answer: {example[1][0]}." for example in examples]
        else:
            sentences = [example[0] for example in examples]

        self.logger.info(f"Starting to use SentenceBERT to encode: bsz={self.index_args.predict_batch_size}")

        # if self.build_index_num_threads == 1:
            # Tokenize sentences

        vectors = sentence_model.encode(
            sentences,
            batch_size=self.index_args.predict_batch_size,
            show_progress_bar=True,
            convert_to_tensor=False, # this will return a stacked tensor, we want default return (list of tensors)
            convert_to_numpy=False,
            normalize_embeddings=True)

        return [vector.detach().cpu().numpy() for vector in vectors]
