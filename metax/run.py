from os import pipe
import wandb
from metax.cli import get_parser
from metax.commons import MetaX_Common, set_seeds
from metax.retrieve import MetaX_Retrieve
# from metax.retrieve_zeroshot import MetaX_RetrieveZeroshot



if __name__ == '__main__':
    args = get_parser().parse_args()

    set_seeds(args.train_seed)

    wandb.init(project="recross", entity="yuchenlin", settings=wandb.Settings(start_method="fork"))
    wandb.config = args
    wandb.run.name = args.run_name
    wandb.run.save()

    pipeline = MetaX_Retrieve(args)
    pipeline.init_logger(args.log_dir)

    if args.action.lower() == 'zero_shot_evaluation':
        pipeline.init_model()
        pipeline.init_target_tasks()
        pipeline.zero_shot_evaluation()
    elif args.action.lower() == 'ret_aug':
        assert args.use_retriever == True
        pipeline.init_model()
        pipeline.init_target_tasks()
        pipeline.init_memory_manager()
        pipeline.unified_pipeline()
    elif args.action.lower() == 'few_shot_evaluation':
        pipeline.init_model()
        pipeline.init_target_tasks()
        if args.use_retriever:
            pipeline.init_memory_manager()  # Updated step.
        else:
            assert args.ret_merge_mode == "none"
        pipeline.unified_pipeline()
    elif args.action.lower().endswith("build_index_cache"):
        assert args.use_retriever
        pipeline.init_memory_manager()
    else:
        raise NotImplementedError(f'Unsupported action {args.action}. Please check your cli arguments')
    pipeline.logger.warning("Done!")
