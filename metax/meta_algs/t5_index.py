
import logging
from tqdm import tqdm
from argparse import ArgumentParser

import numpy as np

from metax.datasets import get_formatted_data
from metax.models import run_bart
from metax.models.mybart import MyBart
from metax.models.utils import (convert_model_to_single_gpu, freeze_embeds, trim_batch)
from metax.task_manager.dataloader import GeneralDataset
from transformers.modeling_bart import _prepare_bart_decoder_inputs
from transformers import T5Tokenizer, T5ForConditionalGeneration

import torch
from torch import Tensor
from torch.nn import functional as F
from torch.nn import Module

from metax.meta_algs.retriever_module import BaseIndexManager

import faiss

def masked_mean(reps, masks):
    masks = masks.view(reps.size()[0], reps.size()[1], 1)
    masked_reps = reps * masks
    masked_reps_sum = masked_reps.sum(dim=1)
    length_reps = masks.sum(dim=1).view(masked_reps_sum.size()[0], 1)
    mean_reps = masked_reps_sum / length_reps
    return mean_reps

class T5IndexManager(BaseIndexManager):

    def __init__(self, logger, args=None):
        super().__init__(logger, args=args)
        self.name = "base_index_manager"

        self.use_cuda = self.index_args.n_gpu > 0

        self.n_gpu = self.index_args.n_gpu

        self.load_upstream_model(model_type='t5-base')  # <-- load_upstream_model()
        self.dim_memory = 2*768 # bart-encoder + bart-decoder
        self.memory_index = None
        self.memory_examples = {}
        self.memory_example_ids = [] # a sorted list which has the same order of the faiss index.

        self.tokenizer = T5Tokenizer.from_pretrained("t5-base")

    def load_upstream_model(self, model_type=None, model_path=None):
        """
        Load model from pretrained
        # TODO: use model_type=facebook/bart-base for testing.
        """
        self.logger.info(f"Loading model {model_type} .....")
        model = T5ForConditionalGeneration.from_pretrained(model_type)
        self.logger.info(f"Loading model {model_type} ..... Done!")

        if self.use_cuda:
            model.to(torch.device("cuda"))
            self.logger.info("Moving to the GPUs.")
            if self.n_gpu > 1:
                model = torch.nn.DataParallel(model)
        self.upstream_model = model

    def memory_encoding(self, examples):
        # Here we use a simple non-trainable way to store the examples.
        bart_rep_vectors = self.get_representation(examples, mode="input")
        curr_tensor = torch.tensor(bart_rep_vectors)
        if self.use_cuda:
            curr_tensor = curr_tensor.to('cuda')
        return curr_tensor

    def query_encoding(self, examples, mode="few-shot"):
        # For few-shot exampels, we have outputs; for zero-shot examples, we don't have outputs.
        # Here we use a simple non-trainable way to encode the query examples.
        bart_rep_vectors = self.get_representation(examples, mode=mode)
        bart_rep_vectors = np.array(bart_rep_vectors)
        vec_shape = bart_rep_vectors.shape

        # need to get the mean of the input/output vectors
        # TODO -- we should discuss if this makes sense. IMO we need different encoder for zero and few shot
        bart_rep_vectors = bart_rep_vectors.reshape(vec_shape[0], int(vec_shape[1]/2), 2)
        query_vector = np.mean(bart_rep_vectors, axis=2)
        curr_tensor = torch.tensor(query_vector)
        if self.use_cuda:
            curr_tensor = curr_tensor.to('cuda')

        return curr_tensor

    def store_examples(self, examples):
        example_ids = []
        for item in examples:
            self.memory_examples[item[2]] = item
            example_ids.append(item[2]) # (input, outputs, id)
        memory_vectors = self.memory_encoding(examples)

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

    def search_index(self, query_vector, k=5):
        assert self.memory_index
        D, I = self.memory_index.search(np.array(query_vector.cpu().detach()), k)
        retrieved_example_ids = [self.memory_example_ids[int(eid)] for eid in I[0]]
        return retrieved_example_ids

    def retrieve_from_memory(self, query_examples, sample_size=5, **kwargs):
        query_vector = self.query_encoding(query_examples)
        retrieved_example_ids = self.search_index(query_vector, sample_size)
        retrieved_examples = [self.memory_examples[rid] for rid in retrieved_example_ids]
        return retrieved_examples

    def update_index(self, example_ids, vectors):
        assert len(example_ids) == len(vectors)
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

    def get_representation(self, examples, mode="input"):
        data_manager = self.get_representation_dataloader(examples)
        all_vectors = []
        t5model = self.upstream_model if (self.n_gpu ==1 or not self.use_cuda) else self.upstream_model.module
        t5model.eval()
        for batch in tqdm(data_manager.dataloader):
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
            encoder_outputs = t5model.model.encoder(
                input_ids, input_attention_mask)
            x = encoder_outputs.last_hidden_state
            x = masked_mean(x, input_attention_mask)

            input_vectors = x.detach().cpu().numpy()

            if mode=="input":
                all_vectors += list(input_vectors)
                continue

            ## Encode the output text with BART-decoder
            output_ids = batch[2]
            output_attention_mask = batch[3]

            decoder_input_ids, decoder_padding_mask, causal_mask = _prepare_bart_decoder_inputs(
                bart_model.model.config,
                input_ids,
                decoder_input_ids=output_ids,
                decoder_padding_mask=output_attention_mask,
                causal_mask_dtype=bart_model.model.shared.weight.dtype,
            )
            decoder_outputs = bart_model.model.decoder(
                decoder_input_ids,
                encoder_outputs[0],
                input_attention_mask,
                decoder_padding_mask,
                decoder_causal_mask=causal_mask,
                decoder_cached_states=None,
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
