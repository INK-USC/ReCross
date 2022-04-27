import copy
from logging import WARN
import torch
import copy
import wandb
import json
import os
import random
from math import ceil

from metax.commons import MetaX_Common
from metax.config import METRICS
from metax.datasets import get_formatted_data, get_all_upstream_data_from_single_file
from metax.meta_algs.biencoder_index import TrainableIndexManager
from metax.meta_algs.retriever_module import RandomMemoryManger
from metax.meta_algs.sentencebase_index import SentenceBartIndexManager
from metax.meta_algs.bartbase_index import BartIndexManager

from itertools import cycle, islice, product
import numpy as np
from scipy.special import softmax


class MetaX_Retrieve(MetaX_Common):

    def init_memory_manager(self):
        self.logger.info("Initializing the Memory Manager modules.")
        retriever_mode = self.args.retriever_mode.lower()
        if retriever_mode[:8] == "twostage":
            _, primary_mode, secondary_mode = retriever_mode.split("-")
            self.logger.info(f"Using twostage index manager ({primary_mode} to {secondary_mode})")
        else:
            primary_mode = retriever_mode
            secondary_mode = None

        self.logger.info(f"Using primary mode index manager {primary_mode}")
        self.memory_manager = self.get_memory_manager(primary_mode)


        if secondary_mode is not None:
            reuse = primary_mode.endswith("bart")                
            self.logger.info(f"Using secondary mode index manager {secondary_mode}")
            self.secondary_memory_manager = self.get_memory_manager(secondary_mode, reuse)

        # Save/Load upstream examples as memory.
        # if self.args.retriever_mode.lower() == 'sentencetransformer':
        cache_path = self.args.memory_cache_path
        cache_loaded = False
        if os.path.exists(cache_path):
            self.logger.info(f"Loading pre-computed memory cache from path {cache_path}.")
            try:
                self.memory_manager.load_memory_from_path(cache_path)
                cache_loaded = True
            except BaseException as ex:
                self.logger.warning(f"Failed to load manager cache from path {cache_path}, exception {ex}.")
        if not cache_loaded:

            # Load all upstream data into manager
            self.logger.info("Reading upstream train data")
            self.upstream_train_data = get_all_upstream_data_from_single_file(self.args.upstream_train_file)
            self.logger.info(f"Loaded {len(self.upstream_train_data)} upstream examples.")

            self.logger.info("Creating and Storing upstream index.")
            # Check whether we should be using only the inputs to build the cache, or inputs and outputs
            mode = "input" if self.args.action.lower() in ["zero_shot_evaluation","zero_shot_build_index_cache","unsupervised"] else "both"
            self.logger.info(f"mode={mode}")
            self.memory_manager.store_examples(self.upstream_train_data, mode)
            self.logger.info("Storing upstream train data... Done!")
            self.memory_manager.save_memory_to_path(cache_path)

        self.logger.info(f"Memory manager Init Done!")
        self.logger.info(f"There are {len(self.memory_manager.memory_examples)} instances in the memory.")

        if self.args.reranker_model_path.strip():
            self.load_reranker()

    def get_memory_manager(self, mode, reuse=False):
        memory_manager = None
        if mode == 'trained':
            self.logger.info("Using trainable index manager")
            # TODO [chrismiller]: Load actual memory from cache
            init_model = None 
            if reuse:
                init_model = self.memory_manager.upstream_model
            memory_manager = TrainableIndexManager(self.logger, self.args, init_model=init_model)
            assert os.path.exists(self.args.memory_encoder_path), f"Memory encoder path {self.args.memory_encoder_path} must exist"
            assert os.path.exists(self.args.query_encoder_path), f"Query encoder path {self.args.query_encoder_path} must exist"

            memory_manager.memory_encoder.load_state_dict(torch.load(self.args.memory_encoder_path))
            memory_manager.query_encoder.load_state_dict(torch.load(self.args.query_encoder_path))

            if self.n_gpu > 0:
                memory_manager.memory_encoder.to("cuda")
                memory_manager.query_encoder.to("cuda")

            self.logger.info("Loaded trainable index manager")
        elif mode == 'sentencetransformer':
            self.logger.info("Using SentenceTransformer index manager")
            memory_manager = SentenceBartIndexManager(self.logger, self.args)
            self.logger.info("Loaded SentenceTransformer index manager")

        elif mode.endswith('bart'):
            self.logger.info("Using BART index manager")
            memory_manager = BartIndexManager(self.logger, self.args)
            self.logger.info("Loaded BART index manager")

        elif mode == 'random':
            self.logger.info("Using random index manager")
            memory_manager = RandomMemoryManger(self.logger)

        return memory_manager

    def load_reranker(self, ):
        self.logger.info(f"Loading reranker from {self.args.reranker_model_path}....")
        from simpletransformers.classification import ClassificationModel, ClassificationArgs
        reranker_args = ClassificationArgs(
            eval_batch_size=200,
            max_seq_length=512,
            fp16=True,
            use_hf_datasets=True,
            do_lower_case=True,
            no_cache=True,
            no_save=True,
            use_multiprocessing=False,
            use_multiprocessing_for_evaluation=False,
            process_count=1
        )
        self.reranker_model = ClassificationModel(
            "roberta", self.args.reranker_model_path, args=reranker_args
        )
        self.logger.info(f"Loading reranker from {self.args.reranker_model_path}.... Done!")


    def get_ret_examples(self, query_examples=[], task_name="", retrieve_seed=None, round_id=-1):
        query_examples = list(query_examples)
        if not retrieve_seed:
            retrieve_seed = self.args.retrieve_seed
        assert self.args.use_retriever and self.args.upstream_num_shots > 0
        if self.args.retriever_mode == "Random":
            # No need to prepare query examples.
            self.logger.info("Using fixed random retrieval")
            self.logger.info(f"# query examples = {len(query_examples)}")

            retrieved_data = \
                self.memory_manager.retrieve_from_memory(
                    sample_size=int(self.args.upstream_num_shots * self.args.reranker_oversample_rate),
                    seed=retrieve_seed,) # Note that the only changable arg is th seed here.
            # path_retrieved_data = f"{self.args.retrieved_data_dir}/Random_retrieved_{self.args.upstream_num_shots}_seed{retrieve_seed}.json"
        else:
            # use a randomly selected 100 examples as the query examples.
            query_sample_size = ceil((self.args.retrieval_sample_percent/100)*len(query_examples))  # default 100
            self.logger.info(f"Using {query_sample_size} datapoints to retrieve")
            if self.args.use_random_retrieval_queries:  # default=False
                self.logger.info("Selecting query examples randomly")
                query_examples = random.sample(query_examples, query_sample_size)
            else:
                self.logger.info("Selecting query ")
                query_examples = query_examples[:query_sample_size]
            self.logger.info(f"# query examples = {len(query_examples)}")
            # we need to retrieve using query examples (few_shot_data)
            mode = "input" # TODO: add both for few-shot later.
            
            
            if self.remove_group != "none":
                # If this is remove group abalation study
                retrieved_data = []
                target_sz = int(self.args.upstream_num_shots * self.args.reranker_oversample_rate)
                retrieve_sz = 2*target_sz
                while len(retrieved_data) < target_sz:
                    # Keep doubling the retrieve size if we can't get enough examples
                    retrieve_sz = min(2*retrieve_sz, 2000000)
                    retrieved_data = self.memory_manager.retrieve_from_memory(
                    query_examples,
                    sample_size=retrieve_sz,
                    mode=mode,
                    seed=retrieve_seed)
                    
                    not_removed = []
                    for dp in retrieved_data:
                        task,_,_,_ = dp[2].split("|")
                        if self.group_of_task[task] != self.remove_group:
                            not_removed.append(dp)
                    retrieved_data = not_removed
                    self.logger.info(f"retrieved {retrieve_sz} examples, after filter left with {len(retrieved_data)}, target size : {target_sz}")
                
                retrieved_data = retrieved_data[:target_sz]
            else:       
                retrieved_data = self.memory_manager.retrieve_from_memory(
                    query_examples,
                    sample_size=int(self.args.upstream_num_shots * self.args.reranker_oversample_rate),
                    mode=mode,
                    seed=retrieve_seed)
            
            # TODO: double check.
            # path_retrieved_data = f"{self.args.retrieved_data_dir}/{self.args.retriever_mode}_retrieved_{task_name}_{self.args.upstream_num_shots}_seed{retrieve_seed}.json"

        
        # TODO: add reranking here and cut-off.
        self.logger.info(f"# retrieved_data （initial) = {len(retrieved_data)}")
        rerank = False
        rerank_mode = "N/A"
        if self.args.reranker_model_path.strip():   # We use this to check if we are going to enable reranking.
            rerank = True
            assert self.reranker_model is not None
            rerank_mode = "Model"

        if hasattr(self, "secondary_memory_manager") and self.secondary_memory_manager is not None:
            rerank = True
            rerank_mode = "Index"

        self.logger.info(f"rerank={rerank};  rerank_mode={rerank_mode}")

        if rerank:
            self.logger.info("Start reranking! ")
            if self.args.reranker_query_size > 0:
                query_examples = query_examples[:self.args.reranker_query_size]

            if rerank_mode == "Index":
                self.secondary_memory_manager.memory_index = None   # clean the memory 
                self.secondary_memory_manager.store_examples(retrieved_data)
                retrieved_data = self.secondary_memory_manager.retrieve_from_memory(query_examples, sample_size=self.args.upstream_num_shots)
            if rerank_mode == "Model":
                packed_pairs = list(product(query_examples, retrieved_data))
                packed_pair_ids = list(product(range(len(query_examples)), range(len(retrieved_data))))
                packed_pairs = [(q[0], c[0]) for q, c in packed_pairs]  # take the inputs only

                self.logger.info(f"# packed_pairs = {len(packed_pairs)}")

                predictions, logits = self.reranker_model.predict(packed_pairs)
                scores = [l[1] for l in logits] # the [0] is the logit for False, [1] is the logit for True

                # aggregate the scores
                final_score_list = {}
                for (qid, cid), score in zip(packed_pair_ids, scores):
                    # final_scores[cid] = max(final_scores[cid], score) # take the maximum score
                    if cid not in final_score_list:
                        final_score_list[cid] = []
                    final_score_list[cid].append(score)
                for cid in final_score_list:
                    # TODO: take max/mean?
                    final_score_list[cid] = float(np.mean(final_score_list[cid]))

                reranked_retrieved_data = [item[0] for item in sorted(zip(retrieved_data, range(len(retrieved_data))), key=lambda x:final_score_list[x[1]], reverse=True)]
                # reranked_retrieved_data = [item[0] for item in sorted(zip(retrieved_data, scores), key=lambda x:x[1], reverse=True)]
                retrieved_data = reranked_retrieved_data[:self.args.upstream_num_shots] # Make sure the size matches

            self.logger.info(f"# retrieved_data (reranked)) = {len(retrieved_data)}")


        path_retrieved_data = f"{self.args.retrieved_data_dir}/{self.args.run_name}_seed={retrieve_seed}_round#{round_id}.json"
        if path_retrieved_data:
            with open(path_retrieved_data, "w") as f:
                f.write(json.dumps(retrieved_data, indent=2))
                self.logger.info(f"Saved retrieved data to {f.name}")
        return retrieved_data


    def unified_pipeline(self, input_model=None):
        self.logger.warning("Unified Pipeline Start...")
        if input_model is None:
            input_model = self.upstream_model

        # Few-shot train & test for multiple rounds
        tasks = {k:v for k, v in self.target_task_train_data.items()}
        self.logger.info(f"Task keys: {tasks.keys()}.")


        for round_id in range(self.args.finetune_round)[:]: # adjust the range
            results = []
            if self.args.retriever_mode == "Random":
                _reused_model = None

            for task_name, multiround_data in tasks.items():
                if self.args.retriever_mode == "Random" and _reused_model:
                    tuned_model = _reused_model # skip re-training across different tasks.
                else:
                    if self.args.num_shots < 16:
                        shots_key_str = "16"
                        self.logger.info(f"using the first {self.args.num_shots} in side the 16-shot list")
                    else:
                        shots_key_str = str(self.args.num_shots)
                    _few_shot_data = multiround_data[shots_key_str][f"round#{round_id}"][:self.args.num_shots]
                    few_shot_data = list(_few_shot_data)
                    # TODO: add an arg to decide what data for early stopping
                    val_dataloader = self.get_dataloader(
                        multiround_data[shots_key_str][f"round#{19}"][:self.args.num_shots], # use last as val
                        mode="eval",
                        task_name=task_name
                    )
                    input_model_copy = copy.deepcopy(input_model)
                    tuned_model = self._subroutine(input_model_copy, few_shot_data, task_name, val_dataloader,
                                            mode=self.args.ret_merge_mode, round_id=round_id)
                    _reused_model = tuned_model

                if self.args.retriever_mode == "Random":
                    few_shot_data = None

                result = self.test_flow(few_shot_data, tuned_model, round_id, task_name=task_name)
                results.append({"task_name": task_name, "result": result})
                wandb.log({"metric": METRICS.get(task_name, 'EM'), "metric_value": result})

            if self.args.retriever_mode == "Random":
                self.args.retrieve_seed += 1 # update the seed for retrieving different random examples across rounds.
                del _reused_model


    def _subroutine(self, upstream_model, few_shot_data, task_name, eval_dataloader, mode="none", round_id=-1):
        # Don't set additional data since that will merge the two datasets
        assert mode in ["none", "mix", "two-stage", "unsupervised"]
        # Override commons.py implementation to use two rounds of training
        assert round_id >= 0

        self.logger.info(f"Using {len(few_shot_data)} in few_shot_data")

        if self.use_cuda:
            upstream_model.to(torch.device("cuda"))
            self.logger.info("Moving to the GPUs.")
            if self.n_gpu > 1:
                self.logger.info(f"Using {torch.cuda.device_count()} GPUS")
                upstream_model = torch.nn.DataParallel(upstream_model)

        additional_data = []
        # This is for the FS+Retriever
        if self.args.use_retriever and self.args.upstream_num_shots > 0:
            query_examples = few_shot_data
            additional_data = self.get_ret_examples(query_examples=query_examples, task_name=task_name, retrieve_seed=self.args.retrieve_seed, round_id=round_id)



        if mode=="none":
            train_data = few_shot_data  # normal few-shot learning
            self.logger.info(f"Using {len(few_shot_data)} (fs) as training data in total. w/ lr = {self.args.learning_rate}")
            self.train_flow(train_data, upstream_model, eval_dataloader, additional_data=None, task_name=task_name, learning_rate=self.args.learning_rate, epochs=self.args.num_train_epochs_fs)
        elif mode == "mix":
            # equalize
            self.logger.info(f"Using {len(few_shot_data)} (fs) + {len(additional_data)} (ret)")
            factor = int(len(additional_data)/len(few_shot_data))
            train_data = few_shot_data * factor + additional_data
            self.logger.info(f"Using {len(few_shot_data)} (fs) * {factor} + {len(additional_data)} (ret) = {len(train_data)} as mixed training data in total. w/ lr = {self.args.learning_rate}")
            self.train_flow(train_data, upstream_model, eval_dataloader, additional_data=None, task_name=task_name, learning_rate=self.args.learning_rate, epochs=self.args.num_train_epochs_fs)
        elif mode == "two-stage":
            assert len(additional_data) > 0
            self.logger.info(f"Using {len(additional_data)} (ret) for 1st stage; w/ lr={self.args.ret_learning_rate} for {self.args.num_train_epochs}")
            upstream_model = self.train_flow(additional_data, upstream_model, eval_dataloader, additional_data=None, task_name=task_name, learning_rate=self.args.ret_learning_rate, epochs=self.args.num_train_epochs)
            self.logger.info(f"Using {len(few_shot_data)} (fs) for 2nd stage; w/ lr={self.args.learning_rate} for {self.args.num_train_epochs_fs}")
            self.train_flow(few_shot_data, upstream_model, eval_dataloader, additional_data=None, task_name=task_name, learning_rate=self.args.learning_rate, epochs=self.args.num_train_epochs_fs)
        elif mode == "unsupervised":    # The new zero-shot + retriever.
            assert len(additional_data) > 0
            if self.args.early_stopping:
                self.logger.info("Creating the split for train/dev to do early stop.")
                ratio = 0.2 # TODO: make it an arg
                dev_num = int(len(additional_data)*ratio)
                additional_data = additional_data[:-dev_num]
                val_data = additional_data[-dev_num:]
                eval_dataloader = self.get_dataloader(val_data, mode="eval", task_name=task_name)
                self.logger.info(f"# additional_data = {len(additional_data)}; # val_data = {len(val_data)}; ")
            self.logger.info(f"Using {len(additional_data)} (ret) for the only unsupervised learning to generalize stage; w/ lr={self.args.ret_learning_rate}")
            self.train_flow(additional_data, upstream_model, eval_dataloader, additional_data=None, task_name=task_name, learning_rate=self.args.ret_learning_rate)
            # no tuning on the few-shot examples at all, only use them as query examples.
        return upstream_model
