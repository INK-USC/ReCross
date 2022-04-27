# Enable import metax
import sys
import os
sys.path.insert(0, "../..")
sys.path.insert(0, ".")

import random
import logging
from argparse import ArgumentParser
from metax.datasets import get_formatted_data
from metax.meta_algs.biencoder_index import TrainableIndexManager
from typing import Optional, List
import torch

from metax.models.utils import pickle_save, pickle_load

class BaseMetaTrainer():

    def __init__(self, logger, meta_args = None, indexer_args = None, query_tasks = [], support_tasks = []):
        super().__init__()
        self.name = "base_meta_trainer"
        self.logger = logger
        self.meta_trainer_args = meta_args
        self.real_upstream_tasks = []   # upstream task names
        self.support_model = None
        self.query_tasks = query_tasks
        self.support_tasks = support_tasks
        self.index_manager = TrainableIndexManager(logger=logger, args=indexer_args)


    def generate_meta_split(self, query_size=5):
        # TODO: split the real_upstream_tasks to query tasks and support tasks
        query_tasks = random.sample(self.real_upstream_tasks, query_size)
        support_tasks = [t for t in self.real_upstream_tasks if t not in query_tasks]
        return query_tasks, support_tasks

    def get_examples(self, task_name, mode):
        assert mode in ["upstream", "fewshot", "eval"]
        # TODO: load the examples
        examples = []
        return examples

    def load_bart_model(self):
        # load a bart model
        return None

    def upstream_train(self, model, examples):
        # train the model with examples
        # use the api here.
        return None

    def get_train_dataloader(self, examples):
        return None

    def get_eval_dataloader(self, examples):
        return None

    def few_shot_finetuning(self, model, examples):
        target_model = copy.deepcopy(model) # do not update the given model in place
        target_dataloader = self.get_train_dataloader(examples)
        # init optimizer, scheduler for few-shot fine-tuning.
        return target_model


    def generate_supervision_for_retriever(self, query_examples, all_retrieved_examples, all_scroes):
        # average the all_scroes
        # rank all_retrieved_examples by all_scores
        # take the top $postive_size as the postive examples
        # take the bottom $negative_size as the negative examples
        # Also, judge the difference between two orders:
            # retreived order
            # score-based order

        return

    def meta_train_step(self):


        # TODO: these can be moved to self.meta_trainer_args
        K = 2048
        k = 16
        num_rounds = 30

        train_data_for_retriever = []
        for query_task in self.query_tasks: # number of the query tasks
            query_eval_examples = self.get_examples(query_task, mode="eval")
            query_fewshot_examples_all = self.get_examples(query_task, mode="few-shot")
            for query_fewshot_examples in all_query_fewshot_examples_all:   # number of different few-shot examples
                # get a certain round of few-shot query examples
                # reference loss
                ref_target_model = self.few_shot_finetuning(self.support_model, query_fewshot_examples)
                ref_loss = self.few_shot_eval(ref_target_model, query_eval_examples, return_loss=True)

                all_retrieved_examples = self.index_manager.retrieve_from_memory(query_examples, sample_size=K)
                all_scroes = {}
                assert len(retrieved_examples) == K
                for round_id in range(num_rounds):  # number of scoring rounds
                    sub_retrieved_examples = random.sample(all_retrieved_examples, k)
                    train_examples = query_examples + sub_retrieved_examples
                    target_model = self.few_shot_finetuning(self.support_model, train_examples)
                    target_loss = self.few_shot_eval(target_model, query_eval_examples)
                    score = ref_loss - target_loss # the higher means the more helpful
                    for ex in sub_retrieved_examples:
                        example_id = ex[2]
                        if example_id not in all_scroes:
                            all_scroes[example_id] = []
                        all_scroes[ex[2]].append(score)
                _supervision = self.generate_supervision_for_retriever(query_examples, all_retrieved_examples, all_scroes)
                train_data_for_retriever.append(_supervision)
        return train_data_for_retriever



    def meta_train_flow(self):
        num_outter_rounds = 10
        num_inner_rounds_max = 30
        for i in range(num_outter_rounds):
            query_tasks, support_tasks = self.generate_meta_split()
            self.query_tasks = query_tasks
            self.support_tasks = support_tasks

            # TODO: use the examples in the self.support_tasks to train a self.support_model
            support_examples = []
            for task in self.support_tasks:
                support_examples += self.get_examples(task, mode="upstream")
            bart_model = self.load_bart_model()
            self.support_model = self.upstream_train(bart_model, support_examples)

            self.index_manager.clean_index()
            self.index_manager.upstream_model = self.support_model # using the current support model.
            self.index_manager.store_memory(support_examples)


            for j in range(num_inner_rounds_max):
                train_data_for_retriever = self.meta_train_step()
                self.index_manager.train_biencoder(train_data_for_retriever)
                # TODO: early quit for this if the retrieved order and the loss-based order are not quite different.


    def pretrain_flow(self, positive_tasks:Optional[List]=None, names:Optional[List] = None, t:str='upstream', num:int=2000):
        support_task_data = get_formatted_data(self.support_tasks)
        query_task_upstream_data = get_formatted_data(self.query_tasks)
        if positive_tasks:
            positive_upstream_data = get_formatted_data(positive_tasks)
        # todo: set num by args
        query_task_data = get_formatted_data(self.query_tasks, t='fewshot', num=16)
        pretrain_data = []
        validation_data = []

        for i,target_task in enumerate(self.query_tasks):
            self.logger.info(f"Processing query task {target_task}")
            fake_upstream_task_data = query_task_upstream_data[target_task]
            query_examples = query_task_data[target_task]
            # filter out actual train examples from the fake positive examples
            train_item_set = set([item[0] for item in query_examples])

            if positive_tasks:
                fake_positive_examples = []
                for key in positive_upstream_data.keys():
                    fake_positive_examples += positive_upstream_data[key]
            else:
                fake_positive_examples = [item for item in fake_upstream_task_data if item[0] not in train_item_set]

            # We don't need to worry about numbers, since the trainer selects a specific number of triplets
            # per query -- must update if not using TripletMarginLoss
            negative_examples = []
            for key in support_task_data.keys():
                negative_examples += support_task_data[key]

            # sample validation
            # TODO: improve efficiency. Can't use set since the items in examples are unhashable lists
            # this is onetime code with relatively short lists so should be okay
            validation_positive = random.sample(fake_positive_examples, k=int(.1*len(fake_positive_examples)))
            fake_positive_examples = [item for item in fake_positive_examples if item not in validation_positive]
            validation_positive = list(validation_positive)

            validation_negative = random.sample(negative_examples, k=int(.1*len(negative_examples)))
            negative_examples = [item for item in negative_examples if item[0] not in validation_negative]
            validation_negative = list(validation_negative)

            self.logger.info(f"Extracted {len(fake_positive_examples)} candidate positive examples, " \
                                + f"{len(negative_examples)} candidate negative examples " \
                                + f"for {len(query_examples)} queries.")

            self.logger.info(f"Found {len(validation_positive)} positive validation examples, " \
                                + f"{len(validation_negative)} negative validation examples ")

            pretrain_data.append((query_examples, fake_positive_examples, negative_examples))
            validation_data.append((query_examples, validation_positive, validation_negative))

        #self.index_manager.train_biencoder(pretrain_data, validation_data=validation_data)
        self.index_manager.memory_encoder.load_state_dict(torch.load("/home/chris/prefixTuning/MetaCross/outputs/memory_encoder_epoch_6.pt"))
        self.index_manager.query_encoder.load_state_dict(torch.load("/home/chris/prefixTuning/MetaCross/outputs/query_encoder_epoch_6.pt"))

        self.index_manager.memory_encoder.to("cuda")
        self.index_manager.query_encoder.to("cuda")
        self.logger.info("Loaded")
        memory_data = []
        for key in support_task_data.keys():
            memory_data += support_task_data[key]
        for key in query_task_upstream_data.keys():
            memory_data += query_task_upstream_data[key]
        self.logger.info("Manufactured")
        self.index_manager.store_memory(memory_data)
        self.logger.info("Stored")
        pickle_save(self.index_manager.memory_index, "/home/chris/prefixTuning/MetaCross/outputs/memory_index")
        pickle_save(self.index_manager.memory_examples, "/home/chris/prefixTuning/MetaCross/outputs/memory_examples")
        pickle_save(self.index_manager.memory_example_ids, "/home/chris/prefixTuning/MetaCross/outputs/memory_example_ids")

    def load_cached_flow(self):
        self.index_manager.memory_encoder.load_state_dict(torch.load("/home/chris/prefixTuning/MetaCross/outputs/memory_encoder_epoch_6.pt"))
        self.index_manager.query_encoder.load_state_dict(torch.load("/home/chris/prefixTuning/MetaCross/outputs/query_encoder_epoch_6.pt"))

        self.index_manager.memory_encoder.to('cuda')
        self.index_manager.query_encoder.to('cuda')
        self.index_manager.memory_index = pickle_load("/home/chris/prefixTuning/MetaCross/outputs/memory_index")
        self.index_manager.memory_examples = pickle_load("/home/chris/prefixTuning/MetaCross/outputs/memory_examples")
        self.index_manager.memory_example_ids = pickle_load("/home/chris/prefixTuning/MetaCross/outputs/memory_example_ids")


if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument("--do_lowercase", action='store_true', default=False,help="train all as lowercase")
    parser.add_argument("--use_cuda", action='store_true', default=False,help="train on GPU")

    # Preprocessing/decoding-related parameters
    parser.add_argument('--max_input_length', type=int, default=128)
    parser.add_argument('--max_output_length', type=int, default=32)
    parser.add_argument('--num_beams', type=int, default=4)
    parser.add_argument('--learning_rate', type=int, default=4)
    parser.add_argument("--append_another_bos",
                        action='store_true',
                        default=False)

    # Training-related parameters
    parser.add_argument("--train_batch_size", default=64, type=int, help="Batch size per GPU/CPU for training.")
    parser.add_argument("--predict_batch_size", default=32, type=int, help="Batch size per GPU/CPU for evaluation.")

    args = parser.parse_args()

    log_filename = "{}log.txt".format("train_")

    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
        datefmt='%m/%d/%Y %H:%M:%S',
        level=logging.INFO,
        handlers=[
            logging.FileHandler(os.path.join('outputs', log_filename)),
            logging.StreamHandler()
        ])
    logger = logging.getLogger(__name__)

    query_tasks = ["cola", "rte", "sentiment140", "cosmos_qa", "common_gen", "wnli", "trivia_qa"]
    #query_tasks = ["qnli"]
    support_tasks = ["coqa", "imdb_reviews", "yelp_polarity_reviews", "copa", "web_nlg_en"]
    positive_tasks = None#["anli_r1", "wnli", "rte"]

    module = BaseMetaTrainer(logger, meta_args=args, indexer_args=args, query_tasks=query_tasks, support_tasks=support_tasks)
    module.pretrain_flow(positive_tasks=positive_tasks)
    module.load_cached_flow()
    target_fewshot_task = "trec"
    query_task_data = get_formatted_data([target_fewshot_task], t='fewshot', num=16)[target_fewshot_task]

    retrieved = module.index_manager.retrieve_from_memory(query_task_data, sample_size=32)
    combined = query_task_data + retrieved
    pickle_save(combined, f"outputs/{target_fewshot_task}_combined_data_2")
