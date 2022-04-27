import copy
from random import shuffle
import torch
from IPython.terminal.embed import embed
from torch.utils import data
from metax.retrieval_based.cli import get_parser
from metax.commons import MetaX_Common, gen, set_seeds


class SimpleRetrievalBasedMethod(MetaX_Common):
    def __init__(self, args):
        super().__init__(args)
        self.args = args
        
    def few_shot_train_and_test(self, input_model=None):
        if self.args.do_retrieve:
            # In case of retrieval based method
            self.logger.warn("Data retrieval start ...")
            generator = gen(self.args.retrieval_seed)

            for idx in range(self.args.num_retrieve_round):
                self.logger.warn(f"---> Data retrieval round #{idx}")
                torch.cuda.empty_cache()
                retrieval_model = copy.deepcopy(self.upstream_model)

                # Construct complete upstream datasets
                upstream_data = []
                for _, task_data in self.upstream_train_data.items():
                    upstream_data += task_data
                
                # Each round takes a different subset of upstream dataset
                gen(generator.integers(65536)).shuffle(upstream_data)
                upstream_data_flat = upstream_data[:self.args.num_retrieval]
                dataloader = self.get_dataloader(upstream_data_flat,
                                                mode='train',
                                                task_name='dummy')[0]

                # Retrain a deepcopy of the model
                self.logger.warn(f"---> Round #{idx} generalization training start ...")
                optimizer, scheduler = self.get_optimizer(retrieval_model, self.args)
                self.model_training(retrieval_model, optimizer, scheduler, dataloader, False)
                self.logger.warn(f"---> Round #{idx} generalization training done")

                # For each round, perform few-shot fine-tune and test
                super().few_shot_train_and_test(retrieval_model)
        else:
            # Simple few-shot fine-tune and test
            super().few_shot_train_and_test(input_model=input_model)


if __name__ == '__main__':
    args = get_parser().parse_args()
    set_seeds(args.seed)

    # Model initialization
    databased = SimpleRetrievalBasedMethod(args)
    databased.init_logger(args.log_dir)
    databased.init_upstream_tasks()
    databased.init_target_tasks()
    databased.init_model()

    # Pipeline
    if args.do_upstream:
        databased.upstream_training()
    databased.few_shot_train_and_test()
