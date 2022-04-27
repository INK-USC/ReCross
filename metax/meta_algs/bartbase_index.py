import logging
from tqdm import tqdm
from argparse import ArgumentParser

import numpy as np

from metax.datasets import get_formatted_data
from metax.models import run_bart
from metax.models.mybart import MyBart
from metax.models.utils import (convert_model_to_single_gpu, freeze_embeds, trim_batch, pickle_save, pickle_load, invert_mask)
from metax.task_manager.dataloader import GeneralDataset
from transformers import BartTokenizer
from collections import defaultdict
import random
import torch
from torch import Tensor
from torch.nn import functional as F
from torch.nn import Module

from metax.meta_algs.retriever_module import BaseIndexManager
import math
import faiss

def masked_mean(reps, masks):
    masks = masks.view(reps.size()[0], reps.size()[1], 1)
    masked_reps = reps * masks
    masked_reps_sum = masked_reps.sum(dim=1)
    length_reps = masks.sum(dim=1).view(masked_reps_sum.size()[0], 1)
    mean_reps = masked_reps_sum / length_reps
    return mean_reps

class BartIndexManager(BaseIndexManager):

    def __init__(self, logger, args=None, init_model=None):
        super().__init__(logger, args=args)
        self.name = "base_index_manager"

        self.use_cuda = self.index_args.n_gpu > 0

        self.n_gpu = self.index_args.n_gpu

        if init_model:
            self.upstream_model = init_model
        else:
            self.init_model()
        self.dim_memory = 1024 if args.ret_merge_mode == "unsupervised" else 2048 #2*768 # bart-encoder + bart-decoder
        self.memory_index = None
        self.memory_examples = {}
        self.memory_example_ids = [] # a sorted list which has the same order of the faiss index.

        self.tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')

    def init_model(self):
        if self.index_args.retriever_mode.lower() == "vbart":
            bart_str = "facebook/bart-large"
        else:
            bart_str = self.index_args.model_type
        self.logger.info(f"Initializing upstream model {bart_str} for indexing module.")
        self.load_upstream_model(model_type=bart_str)

    def load_upstream_model(self, model_type=None, model_path=None):
        """
        Load model from pretrained or checkpoints
        """
        if model_path:
            self.logger.info(f"Loading checkpoint from {model_path} for {model_type} .....")
            model = MyBart.from_pretrained(model_type,
                                           state_dict=convert_model_to_single_gpu(
                                               torch.load(model_path)))
            self.logger.info(f"Loading checkpoint from {model_path} for {model_type} ..... Done!")
        else:
            self.logger.info(f"Loading model {model_type} .....")
            model = MyBart.from_pretrained(model_type)
            self.logger.info(f"Loading model {model_type} ..... Done!")
        if self.use_cuda:
            if self.n_gpu > 1:
                self.logger.info(f"Setting as dataparallel for {self.n_gpu} gpus.")
                model = torch.nn.DataParallel(model)
            model.to(torch.device("cuda"))
            self.logger.info("Moving to the GPUs.")
        self.upstream_model = model

    def memory_encoding(self, examples, mode="both", silence_tqdm=False):
        # Here we use a simple non-trainable way to store the examples.
        bart_rep_vectors = self.get_representation(examples, mode=mode, silence_tqdm=silence_tqdm)
        curr_tensor = torch.tensor(bart_rep_vectors)
        if self.use_cuda:
            curr_tensor = curr_tensor.to('cuda')
        return curr_tensor

    def query_encoding(self, examples, mode="both", silence_tqdm=False):
        # For few-shot examples, we have outputs; for zero-shot examples, we don't have outputs.
        # Here we use a simple non-trainable way to encode the query examples.
        bart_rep_vectors = self.get_representation(examples, mode=mode, silence_tqdm=silence_tqdm)
        curr_tensor = torch.tensor(bart_rep_vectors)
        if self.use_cuda:
            curr_tensor = curr_tensor.to('cuda')

        return curr_tensor

    def store_examples(self, examples, mode="input"):
        self.logger.info(f"All #examples={len(examples)}")
        # Splitting for Parallelizing
        n_shards =  self.index_args.num_shards_for_indexing
        shard_id = self.index_args.shard_id_for_indexing
        # use tolist to make the items serializable
        examples = [item.tolist() for item in np.array_split(examples, n_shards)[shard_id]]

        self.logger.info(f"n_shards={n_shards}; shard_id={shard_id}; #examples={len(examples)}")


        example_ids = []
        for item in examples:
            self.memory_examples[item[2]] = item
            example_ids.append(item[2]) # (input, outputs, id)

        memory_vectors = self.memory_encoding(examples, mode)

        self.logger.info(f"Stored {len(memory_vectors)} vectors (for {len(examples)} examples)")
        assert len(memory_vectors) == len(examples)
        self.logger.info(f"Stored vectors with dim {len(memory_vectors[0])}.")
        assert len(memory_vectors[0]) == self.dim_memory
        self.update_index(example_ids, memory_vectors)

    def clean_index(self):
        del self.memory_index
        self.memory_index = None
        self.memory_examples = {}
        self.memory_example_ids = []

    def search_index(self, query_vector, k=8, expand=False):
        assert self.memory_index
        qry = np.array(query_vector.cpu().detach())
        distances, indices = self.memory_index.search(qry, k)
        if expand:
            # return full info for all queries
            return (distances, indices)
        retrieved_example_ids = []
        for result in indices:
            retrieved_example_ids += [self.memory_example_ids[int(eid)] for eid in result]

        return retrieved_example_ids

    def retrieve_from_memory(self, query_examples, sample_size=512, seed=None, mode="input", **kwargs):
        if self.index_args.query_aggregation_mode == "aggregate_scores":
            query_vector = self.query_encoding(query_examples, mode)
            ur = 1.5
            # double sample size to give more overlap in retrieved items
            each_sample_size = math.ceil(sample_size/len(query_examples)*ur) 
            self.logger.info(f"aggregate_scores: each_sample_size={each_sample_size}")
            distances,indices = self.search_index(query_vector, each_sample_size, expand=True)
            aggregate_distances = defaultdict(int)
            occurrences = defaultdict(int)
            for distance_vec, index_vec in zip(distances, indices):
                for distance, index in zip(distance_vec, index_vec):
                    if index == -1:
                        # if there are not enough items, faiss pads with -1
                        continue
                    aggregate_distances[index] += distance
                    occurrences[index] += 1
            for key in occurrences.keys():
                # TODO -- this does not penalize examples which are NOT in the first 2*sample_size results for all queries. Need to fix.
                aggregate_distances[key] /= occurrences[key]

            # select top sample_size (ie, lowest distances)
            retrieved_example_ids = [self.memory_example_ids[x[0]] for x in sorted(aggregate_distances.items(), key=lambda kv: kv[1])][:sample_size]
            retrieved_examples = [self.memory_examples[rid] for rid in retrieved_example_ids]
        elif self.index_args.query_aggregation_mode == "aggregate_choices":
            query_vector = self.query_encoding(query_examples, mode)

            each_sample_size = math.ceil(sample_size/len(query_examples))
            self.logger.info(f"aggregate_choices: each_sample_size={each_sample_size}")
            distances, indices = self.search_index(query_vector, each_sample_size, expand=True)
            retrieved_examples = [self.memory_examples[self.memory_example_ids[rid]] for sublist in indices for rid in sublist]
            assert len(retrieved_examples) == sample_size
        else:
            raise ValueError(f"Invalid query aggregation mode {self.args.query_aggregation_mode}")

        return retrieved_examples


    def retrieve_from_memory_v1(self, query_examples, sample_size=512, seed=None, mode="input", **kwargs):
        query_vector = self.query_encoding(query_examples, mode)
        self.logger.info(f"Retrieving for {len(query_examples)} examples, encoded to dim {query_vector.shape}.")

        k = int(sample_size/len(query_examples))
        retrieved_example_ids = self.search_index(query_vector, k)
        retrieved_examples = [self.memory_examples[rid] for rid in retrieved_example_ids]

        return retrieved_examples

    def update_index(self, example_ids, vectors):
        assert len(example_ids) == len(vectors)

        if self.index_args.num_shards_for_indexing > 1:
            # do not save the faiss index now, we merge the vectors and do the indexing later.
            self.memory_example_ids += example_ids
            self.memory_index = np.array(vectors.cpu().detach())
            return

        if not self.memory_index:
            # init the faiss index
            self.memory_index = faiss.IndexFlatL2(self.dim_memory)
        self.memory_example_ids += example_ids
        vectors = np.array(vectors.cpu().detach())
        self.memory_index.add(vectors)

    def get_representation_dataloader(self, formatted_data):
        rep_dataloader = GeneralDataset(
            self.logger,
            self.index_args,
            None,
            is_training=True,  # Random Sampler
            task_name="task",
            given_data=formatted_data)
        rep_dataloader.load_dataset(self.tokenizer, skip_cache=True, quiet=True)
        # get the sequential sampler + train_data type (input &  output)
        rep_dataloader.load_dataloader(is_training=False)
        return rep_dataloader

    def get_representation(self, examples, mode="both", silence_tqdm = False):
        self.logger.debug(f"Getting representation for {len(examples)} examples, mode={mode}.")
        data_manager = self.get_representation_dataloader(examples)
        all_vectors = []
        bart_model = self.upstream_model if (self.n_gpu ==1 or not self.use_cuda) else self.upstream_model.module
        bart_model.eval()
        for batch in tqdm(data_manager.dataloader, disable=silence_tqdm):
            if self.use_cuda:
                batch = [b.to(torch.device("cuda")) for b in batch]
            pad_token_id = self.tokenizer.pad_token_id
            batch[0], batch[1] = trim_batch(
                batch[0], pad_token_id, batch[1])
            batch[2], batch[3] = trim_batch(
                batch[2], pad_token_id, batch[3])

            ## Encode the input text with BART-encoder
            input_ids = batch[0]
            input_attention_mask = batch[1]
            encoder_outputs = bart_model.model.encoder(
                input_ids, input_attention_mask)
            x = encoder_outputs[0]
            # TODO [chrismiller]: Make this configurable via args (using this vs
            # using x = x[:, 0, :]
            x = masked_mean(x, input_attention_mask)

            input_vectors = x.detach().cpu().numpy()

            if mode=="input":
                all_vectors += list(input_vectors)
                continue

            ## Encode the output text with BART-decoder
            decoder_input_ids = batch[2]
            decoder_mask = invert_mask(batch[3])
            if decoder_mask is not None and decoder_mask.shape[1] > 1:
                decoder_mask[:, 0] = decoder_mask[:, 1]

            decoder_outputs = bart_model.model.decoder(
                input_ids=decoder_input_ids,
                attention_mask=decoder_mask,
                encoder_hidden_states=encoder_outputs[0],
                encoder_attention_mask=input_attention_mask,
                use_cache=False
            )

            y = decoder_outputs[0]
            y = y[:, 0, :]
            output_vectors = y.detach().cpu().numpy()

            del batch
            del encoder_outputs
            del decoder_outputs
            # concatenate the vectors
            vectors = np.concatenate([input_vectors, output_vectors], axis=1)
            all_vectors += list(vectors)
        return all_vectors

    def load_memory_from_path(self, init_memory_cache_path):
        self.logger.info("Loading cached index manager memory")
        cache = pickle_load(init_memory_cache_path)
        self.dim_memory = cache["dim_memory"]
        self.memory_index = cache["memory_index"]
        self.memory_examples = cache["memory_examples"]
        self.memory_example_ids = cache["memory_example_ids"]
        self.logger.info("Loading cached index manager memory... Done!")

    def save_memory_to_path(self, memory_pkl_path):
        try:
            self.logger.info("Caching index manager memory")
            cache = {}
            cache["dim_memory"] = self.dim_memory
            cache["memory_index"] = self.memory_index
            cache["memory_examples"] = self.memory_examples
            cache["memory_example_ids"] = self.memory_example_ids
            pickle_save(cache, memory_pkl_path)
            self.logger.info("Caching index manager memory... Done!")
        except BaseException as ex:
            self.logger.warning(f"Failed to cache manager, exception {ex}.")
