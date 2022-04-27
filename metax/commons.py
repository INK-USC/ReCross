"""
This script defines the key functions to complete for a data/retrieval-based meta-cross method.
"""
import copy
import json
import logging
import os
import random
import time
from datetime import datetime
from logging import INFO, WARN

import numpy as np
import torch
import wandb
from transformers import AutoTokenizer, BartConfig

from metax.cli import get_parser
from metax.datasets import get_formatted_data
from metax.models.mybart import MyBart
from metax.models.myt5 import MyT5
from metax.models.utils import convert_model_to_single_gpu
from metax.task_manager.dataloader import GeneralDataset
from metax.task_manager.eval_metrics import METRICS
from metax.utils import get_optimizer, metric_evaluate, model_train
from metax.config import UPSTREAM_DATASETS, DOWNSTREAM_DATASETS, DOWNSTREAM_DATASETS_foldernames, FULL_TARGET_TASKS_FOR_T0


os.environ['TOKENIZERS_PARALLELISM'] = 'False'

class MetaX_Common():

    def __init__(self, args=None):
        self.args = args  # the configurations

        # Training settings
        self.num_upstream_samples = self.args.num_upstream_samples

        self.upstream_tasks = UPSTREAM_DATASETS
        self.target_tasks = FULL_TARGET_TASKS_FOR_T0 if len(self.args.target_task) == 0 or self.args.target_task == 'all' else [self.args.target_task]

        # Load task grouping 
        grouping = json.load(open(self.args.grouping_config_file))
        self.group_of_task = {}
        for group, datasets in grouping.items():
            for d in datasets:
                self.group_of_task[d] = group
        # Check whether remove_group is a valid group name
        self.remove_group = self.args.remove_group
        if self.remove_group != "none":
            assert self.remove_group in grouping.keys(), f"Group to remove: {self.remove_group} is not a valid group name."
        
        # Evaluation metric
        # self.metric = 'EM'

        self.use_cuda = torch.cuda.is_available()
        if self.use_cuda:
            self.n_gpu = torch.cuda.device_count()
        else:
            self.n_gpu = 0
        self.logger = None 

    def init_logger(self, logging_dir=None, log_level=INFO):
        logger = logging.getLogger()
        formatter = logging.Formatter("[%(asctime)s][%(levelname)s\t] %(message)s",
                                      datefmt='%Y-%m-%d %H:%M:%S')
        logger.setLevel(log_level)

        ch = logging.StreamHandler()
        ch.setLevel(log_level)
        ch.setFormatter(formatter)
        logger.addHandler(ch)

        if logging_dir is not None:
            os.makedirs(logging_dir, exist_ok=True)
            now = int(round(time.time() * 1000))
            timestr = \
                time.strftime('%Y-%m-%d_%H-%M', time.localtime(now / 1000))
            filename = \
                os.path.join(logging_dir, f"{self.args.run_name}.log")
            fh = logging.FileHandler(filename)
            fh.setLevel(log_level)
            fh.setFormatter(formatter)
            logger.addHandler(fh)
        self.logger = logger

        # Print cli arguments at initialization
        for k, v in vars(get_parser().parse_args()).items():
            self.logger.log(INFO, f"{k:>30} ====> {v}")
        self.logger.info(
            f"Using {self.args.n_gpu} gpus. Train batch size per gpu is {self.args.train_batch_size}.  "
                + f"# gradient_accumulation_steps = {self.args.gradient_accumulation_steps} "
                + f"Overall train batch size is {self.args.train_batch_size * self.args.n_gpu} ")
        self.logger.info(f"Number of upstream samples: {self.num_upstream_samples}")
        self.logger.info(f"Number of shots of fine-tune: {self.args.num_shots}")
        self.logger.info(f"Number of rounds of fine-tune: {self.args.finetune_round}")

        return self.logger 

    def init_upstream_tasks(self, mode='all'):
        """
        Prepare training samples for upstream learning
        Each sample has an input string and an output string
        """
        if not mode == 'all':
            self.upstream_train_data = get_formatted_data(self.upstream_tasks, 'upstream',
                                                        self.num_upstream_samples)
            upstream_train_data = []
            for _, task_data in self.upstream_train_data.items():
                upstream_train_data += task_data
            self.upstream_train_data = upstream_train_data
            self.logger.error(f"Total size of upstream data: {len(upstream_train_data)}")

        self.upstream_validation_data = get_formatted_data(self.upstream_tasks, 'validation', 1000)
        upstream_validation_data = []
        for _, task_data in self.upstream_validation_data.items():
            upstream_validation_data += task_data
        self.upstream_validation_data = upstream_validation_data
        self.logger.error(f"Total size of upstream data: {len(upstream_validation_data)}")

        self.upstream_test_data = get_formatted_data(self.upstream_tasks, 'test', 1000)

    def init_target_tasks(self):
        """
        Prepare training samples for downstream fine-tuning
        Generate samples needed for multiple runs
        """
        self.target_task_train_data = get_formatted_data(self.target_tasks, 'fewshot')
        self.target_task_test_data = get_formatted_data(self.target_tasks, 'test', 1000)

    def get_dataloader(self, formatted_data, mode="both", task_name=None):
        if mode == "train":
            # for training
            data_loader = GeneralDataset(
                self.logger,
                self.args,
                None,
                is_training=True,  # Random Sampler
                task_name=task_name,
                given_data=formatted_data)
            data_loader.load_dataset(self.tokenizer, skip_cache=True, quiet=True)
            data_loader.load_dataloader()
        elif mode == "eval":
            # for evaluation
            data_loader = GeneralDataset(
                self.logger,
                self.args,
                None,
                is_training=False,  # Sequential Sampler
                task_name=task_name,
                given_data=formatted_data)
            # New: use a different max input len for inference which can be diff. from training time.
            data_loader.max_input_length = self.args.max_input_length_for_eval
            data_loader.load_dataset(self.tokenizer, skip_cache=True, quiet=True)
            data_loader.load_dataloader()
        else:
            raise NotImplementedError("unsupported mode")

        return data_loader

    def init_model(self):
        
        model_type, model_path = self.args.model_type, self.args.model_path
        
        if model_path:
            self.logger.info(f"Loading checkpoint from {model_path} for {model_type} .....")

            if 'BART' in self.args.model_type.upper():
                model = MyBart.from_pretrained(model_type,
                                            state_dict=convert_model_to_single_gpu(
                                                torch.load(model_path, map_location=torch.device('cpu'))['model_state']))
            elif 'T0' in self.args.model_type or 'T5' in self.args.model_type:
                model = MyT5.from_pretrained(model_type,
                                            state_dict=convert_model_to_single_gpu(
                                                torch.load(model_path, map_location=torch.device('cpu'))['model_state']))
            self.logger.info(f"Loading checkpoint from {model_path} for {model_type} ..... Done!")
        else:
            self.logger.info(f"Loading model {model_type} .....")
            if 'BART' in self.args.model_type.upper():
                config = BartConfig.from_pretrained(model_type)
                # config.forced_bos_token_id = False
                model = MyBart.from_pretrained(model_type, config=config)
            elif 'T0' in self.args.model_type or 'T5' in self.args.model_type:
                model = MyT5.from_pretrained(model_type)
                # model = T5ForConditionalGeneration.from_pretrained(model_type)
            else:
                self.logger.info(f"model name unmatched")
            assert model is not None
            self.logger.info(f"Loading model {model_type} ..... Done!")

        self.upstream_model = model
        self.tokenizer = AutoTokenizer.from_pretrained(self.args.model_type)
        self.logger.warning('Initialization completed!')

    def upstream_evaluation(self):
        """upstream model inference and evaluate with corresponding metrics"""
        for task_name, task_data in self.upstream_test_data.items():
            upstream_test_dataloader = self.get_dataloader(
                task_data,
                mode="eval",
                task_name=task_name,
            )
            predictions, result = metric_evaluate(
                self.upstream_model,
                upstream_test_dataloader,
                save_predictions=True,
                args=self.args,
                logger=self.logger,
            )
            self.logger.log(WARN, f"Evaluate {task_name} with {METRICS.get(task_name, 'EM')}: {result}")

    def zero_shot_evaluation(self):
        if self.use_cuda:
            if self.n_gpu > 1:
                self.upstream_model = torch.nn.DataParallel(self.upstream_model)
            self.upstream_model.to(torch.device("cuda"))
            self.logger.info("Moving to the GPUs.")
        for task_name, task_data in self.target_task_test_data.items():
            upstream_test_dataloader = self.get_dataloader(
                task_data,
                mode="eval",
                task_name=task_name,
            )
            predictions, result = metric_evaluate(
                self.upstream_model,
                upstream_test_dataloader,
                save_predictions=True,
                args=self.args,
                logger=self.logger,
                prefix=self.args.prefix
            )
            self.logger.log(WARN, f"Evaluate {task_name} : {result}")

    def train_flow(self, train_data, upstream_model, eval_dataloader, additional_data = None, task_name="", saving_dir=None, checkpoint_prefix="fewshot-train", learning_rate=None, epochs=1):
        if additional_data:
            train_data += additional_data

        train_dataloader = self.get_dataloader(train_data,
                                                mode="train",
                                                task_name=task_name)
        total_steps = self.args.total_steps
        if total_steps <= 0:
            total_steps = epochs * len(train_dataloader.dataloader) // self.args.gradient_accumulation_steps
            self.logger.info(f"total_steps={total_steps} <--- computed")
        if learning_rate is None:
            learning_rate = self.args.learning_rate
        optimizer, scheduler = get_optimizer(
            upstream_model,
            learning_rate=learning_rate,
            weight_decay=self.args.weight_decay,
            adam_epsilon=self.args.adam_epsilon,
            warmup_steps=self.args.warmup_steps,
            total_steps=total_steps,
        )
        self.logger.info(f"Init the optimizer with lr={learning_rate}")
        save_model = True if saving_dir else False
        model_train(
            upstream_model,
            optimizer,
            train_dataloader,
            run_name=checkpoint_prefix,
            loss_evaluation=True,
            evaluate_dataloader=eval_dataloader,
            evaluation_steps=1, # self.args.evaluation_steps,
            total_steps=total_steps,
            gradient_steps=self.args.gradient_accumulation_steps,
            saving_steps=self.args.saving_steps,
            early_stop=self.args.early_stopping,
            num_epochs=epochs,
            gpus=list(range(self.n_gpu)),
            scheduler=scheduler,
            logger=self.logger,
            unfreezed_layers=self.args.finetune_layers,
            save_model=save_model,
            saving_dir=saving_dir
        )
        return upstream_model

    def test_flow(self, train_data, upstream_model, round_id=-1, task_name="", test_data=None):
       
        if not test_data:
            test_data = self.target_task_test_data[task_name]

        test_dataloader = self.get_dataloader(
            test_data,
            mode="eval",
            task_name=task_name,
        )
        test_predictions, test_result = metric_evaluate(
            upstream_model,
            test_dataloader,
            save_predictions=True,
            args=self.args,
            logger=self.logger,
            prefix=f'{task_name}-test-round#{round_id}',
        )
        self.logger.info(f"test_perf: {task_name} round #{round_id} with {METRICS.get(task_name, 'EM')}--> {test_result}")

        if train_data:
            train_eval_dataloader = self.get_dataloader(
                train_data,
                mode="eval",
                task_name=task_name,
            )
            train_predictions, train_result = metric_evaluate(
                upstream_model,
                train_eval_dataloader,
                save_predictions=True,
                args=self.args,
                logger=self.logger,
                prefix=f'{task_name}-train-round#{round_id}',
            )
            self.logger.info(f"train_predictions[:16]:{train_predictions[:16]}")
            self.logger.info(f"train_perf: {task_name} round #{round_id} with {METRICS.get(task_name, 'EM')}--> {train_result}")

        return test_result


def set_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def gen(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed=seed)
