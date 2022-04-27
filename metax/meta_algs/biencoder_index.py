# Enable import metax
import sys
import os
import json
sys.path.insert(0, "../..")

import logging
from tqdm import tqdm
from argparse import ArgumentParser

import random

import numpy as np

from typing import Tuple, List, Optional

from metax.datasets import get_formatted_data
from metax.models import run_bart
from metax.models.mybart import MyBart
from metax.models.utils import (convert_model_to_single_gpu, freeze_embeds, trim_batch)
from metax.task_manager.dataloader import GeneralDataset
from transformers import BartTokenizer

import torch
from torch import Tensor
from torch.nn import functional as F
from torch.nn import Module

from metax.meta_algs.retriever_module import BaseIndexManager
from metax.meta_algs.bartbase_index import BartIndexManager

class TrainableIndexManager(BartIndexManager):

    def __init__(self, logger, args=None, input_size=768, output_size=128, init_model=None):
        super().__init__(logger, args, init_model=init_model)  # This part will set up the self.dim_memory as 1024/2048
        self.name = "mlp_biencoder_index_manager"
        self.memory_encoder = MLP(self.dim_memory, output_size)
        self.query_encoder = MLP(self.dim_memory, output_size)
        self.logger.info(f"DIM: {self.dim_memory}")
        self.dim_memory = output_size # this is used for building index

    def train_biencoder(self, train_data, validation_data):
        lr = self.index_args.learning_rate
        memory_optimizer = torch.optim.Adam(self.memory_encoder.parameters(), lr=lr)
        query_optimizer = torch.optim.Adam(self.query_encoder.parameters(), lr=lr)
        memory_scheduler = torch.optim.lr_scheduler.StepLR(memory_optimizer, 8, gamma=0.1, last_epoch=-1)
        query_scheduler = torch.optim.lr_scheduler.StepLR(query_optimizer, 8, gamma=0.1, last_epoch=-1)

        n_epochs = self.index_args.n_epochs

        self.memory_encoder.train()
        self.query_encoder.train()
        if self.use_cuda:
            self.memory_encoder.to('cuda')
            self.query_encoder.to('cuda')

        blop = False
        best_validation_acc = 0
        mode= "input" if self.index_args.ret_merge_mode == "unsupervised" else "both"

        for epoch in range(n_epochs):
            validation_acc = self.validate(validation_data)
            self.logger.info(f"Validation acc: {validation_acc}.")
            if validation_acc > best_validation_acc:
                self.logger.info(f"New best validation acc {validation_acc} at epoch {epoch}.")
                best_validation_acc  = validation_acc
                torch.save(self.memory_encoder.state_dict(), os.path.join("outputs", f"memory_encoder_epoch_{epoch}.pt"))
                torch.save(self.query_encoder.state_dict(), os.path.join("outputs", f"query_encoder_epoch_{epoch}.pt"))

            self.logger.info(f"Training epoch {epoch}/{n_epochs}")
            losses = []
            memory_optimizer.zero_grad()
            query_optimizer.zero_grad()

            n_steps = 128
            n_neg = 8
            batch_size=4
            for step in range(n_steps):
                batch = random.sample(train_data, k=min(batch_size, len(train_data)))
                (queries, candidates, targets) = create_batch_from_groups(batch, neg_size=n_neg)

                query_inputs = self.normalize(self.query_encoding(queries, mode=mode, silence_tqdm = False))
                memory_inputs = self.normalize(self.memory_encoding(candidates, mode=mode, silence_tqdm = False))

                # scores: (k x k*[1+n_neg])
                scores = torch.matmul(query_inputs, memory_inputs.transpose(0, 1))

                # targets: (k,)
                target_tensor = torch.LongTensor(targets)
                if self.use_cuda:
                    target_tensor = target_tensor.to('cuda')
                loss = F.cross_entropy(scores, target_tensor, reduction="mean")

                losses.append(loss.item())
                self.logger.info(f"Loss: {loss.item()}")
                loss.backward()

                memory_optimizer.step()
                query_optimizer.step()

            self.logger.info(f"Completed epoch with avg training loss {sum(losses)/len(losses)}.")
            memory_scheduler.step()
            query_scheduler.step()

    def memory_encoding(self, examples, mode="input", silence_tqdm=False):
        bart_representation = super().memory_encoding(examples, mode=mode, silence_tqdm=silence_tqdm)
        return self.memory_encoder(bart_representation)

    def query_encoding(self, examples, mode="input", silence_tqdm=False):
        bart_representation = super().query_encoding(examples, mode=mode, silence_tqdm=silence_tqdm)
        return self.query_encoder(bart_representation)

    def validate(self, validation_data, k=200, verbose=False):
        accs = []
        totalMatches = 0
        totalK = 0
        for i,(query_examples, validation_positive, validation_negative) in enumerate(validation_data[:500]):
            k = len(validation_positive)
            validation_examples = validation_positive + validation_negative
            mode= "input" if self.index_args.ret_merge_mode == "unsupervised" else "both"
            query_inputs = self.normalize(self.query_encoding(query_examples, mode=mode, silence_tqdm=True))
            memory_inputs = self.normalize(self.memory_encoding(validation_examples, mode=mode, silence_tqdm=True))

            scores = torch.matmul(query_inputs, memory_inputs.transpose(0, 1))

            # We want a single score for each example, so just avg over queries
            scores = scores.mean(0)
            pos_mean = scores[0:len(validation_positive)].mean()
            neg_mean = scores[len(validation_positive):].mean()
            if verbose: self.logger.info(f"Found mean score of {pos_mean} for positive, {neg_mean} for negative.")
            topk_indices = torch.topk(scores, k=k)[1]
            topk_set = {x.item() for x in topk_indices}

            targets = set(range(len(validation_positive)))
            matches = k - len(topk_set - targets)

            if verbose: self.logger.info(f"Evaluating {len(validation_positive)} positive, {len(validation_negative)} negative.")
            if verbose: self.logger.info(f"Found {matches} positive examples in top {k} ({matches*100/k}% of top k).")

            rand_proportion = int(k*(len(validation_positive)/(len(validation_negative)+len(validation_positive))))
            if verbose: self.logger.info(f"(vs {rand_proportion} expected in a random sample).")

            accs.append(matches*100/k)
        return sum(accs)/len(accs)

    def normalize(self, inputs):
        return F.normalize(inputs, dim=0)


def create_batch_from_groups(groups, neg_size=1):
    queries, candidates, targets = [], [], []
    for i,group in enumerate(groups):
        # Correct #
        queries.append(random.choice(group[0]))
        candidates.append(random.choice(group[1]))
        candidates += random.choices(group[2], k=neg_size)

        # The positive candidate is at this point
        target_index = i*(1+neg_size)
        targets.append(target_index)

    assert len(queries) == len(groups)
    assert len(candidates) == len(groups) * (1+neg_size)
    return queries, candidates, targets#np.array(queries), np.array(candidates), np.array(targets)

class MLP(Module):
    def __init__(self, n_inputs, n_outputs):
        super().__init__()
        self.layers = torch.nn.Sequential(
          torch.nn.Flatten(),
          torch.nn.Linear(n_inputs, n_outputs*2),
          torch.nn.ReLU(),
          torch.nn.Linear(n_outputs*2, n_outputs)
        )

    def forward(self, X):
        X = self.layers(X)
        return X
