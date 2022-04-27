# Enable import metax
import sys
import os
sys.path.insert(0, "../..")
import numpy as np
import torch

from transformers import BartTokenizer, BartConfig
from transformers import AdamW, get_linear_schedule_with_warmup

from metax.task_manager.dataloader import GeneralDataset

from .mybart import MyBart
from .utils import freeze_embeds, trim_batch, convert_model_to_single_gpu
import json

from tqdm import tqdm
import copy

def get_override_dataloader(formatted_data, args, logger, tokenizer):
    rep_dataloader = GeneralDataset(
        logger,
        args,
        None,
        data_type="train",
        is_training=True,  # Random Sampler
        task_name="task",
        given_data=formatted_data)
    rep_dataloader.load_dataset(tokenizer, skip_cache=True, quiet=True)
    # get the sequential sampler + train_data type (input &  output)
    rep_dataloader.load_dataloader(is_training=False)
    return rep_dataloader

def run(args, logger, given_data=None):
    tokenizer = BartTokenizer.from_pretrained("facebook/bart-large")

    dev_data = GeneralDataset(logger, args, args.dev_file,
                              data_type="dev", is_training=False, task_name=args.dataset)

    if given_data is None:
        logger.info("Using given data")
        train_data = GeneralDataset(logger, args, args.train_file,
                                    data_type="train", is_training=True, task_name=args.dataset)
        train_data.load_dataset(tokenizer)
        train_data.load_dataloader()
    else:
        logger.info("Using given data")
        train_data = get_override_dataloader(given_data, args, logger, tokenizer)

    dev_data.load_dataset(tokenizer, skip_cache=True)
    dev_data.load_dataloader()

    best_dev_performance = None
    test_performance = None

    best_model_state_dict = None

    if args.do_train:
        if args.checkpoint is not None and args.checkpoint != "None":
            model = MyBart.from_pretrained(args.model,
                                           state_dict=convert_model_to_single_gpu(torch.load(args.checkpoint)))
        else:
            model = MyBart.from_pretrained(args.model)

        if args.freeze_embeds:
            logger.info("Freezing embeddings")
            freeze_embeds(model)

        if args.n_gpu > 1:
            model = torch.nn.DataParallel(model)

        if torch.cuda.is_available():
            model.to(torch.device("cuda"))

        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if not any(
                nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
            {'params': [p for n, p in model.named_parameters() if any(
                nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]

        args.total_steps = args.num_train_epochs * len(train_data.dataloader)
        logger.info(f"args.total_steps = {args.total_steps}")
        optimizer = AdamW(optimizer_grouped_parameters,
                          lr=args.learning_rate, eps=args.adam_epsilon)
        scheduler = get_linear_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=args.warmup_steps,
                                                    num_training_steps=args.total_steps)
        best_dev_performance, best_model_state_dict = train(
            args, logger, model, train_data, dev_data, optimizer, scheduler)

    if args.do_predict:
        if args.do_train and best_model_state_dict is not None:
            model = MyBart.from_pretrained(args.model,
                                           state_dict=best_model_state_dict)
            logger.info("Loading checkpoint from CPU")
        else:
            checkpoint = os.path.join(args.predict_checkpoint)
            model = MyBart.from_pretrained(args.model,
                                           state_dict=convert_model_to_single_gpu(torch.load(checkpoint)))
            logger.info("Loading checkpoint from {}".format(checkpoint))

        if torch.cuda.is_available():
            model.to(torch.device("cuda"))
        model.eval()

        data_type = "test" if "test" in args.test_file else "dev"
        test_data = GeneralDataset(
            logger, args, args.test_file, data_type=data_type, is_training=False, task_name=args.dataset)

        test_data.load_dataset(tokenizer)
        test_data.load_dataloader()

        test_performance = inference(
            model, test_data, save_predictions=True, verbose=True, args=args, logger=logger)
        logger.info("%s on %s data: %.s" % (test_data.metric,
                    test_data.data_type, str(test_performance)))

    return best_dev_performance, test_performance


def train(args, logger, model, train_data, dev_data, optimizer, scheduler):
    model.train()
    global_step = 0
    train_losses = []
    best_performance = None
    stop_training = False

    logger.info("Starting training!")
    for epoch in range(int(args.num_train_epochs)):
        for batch in tqdm(train_data.dataloader, desc="Epoch {}".format(epoch), disable=args.quiet):
            global_step += 1
            if torch.cuda.is_available():
                batch = [b.to(torch.device("cuda")) for b in batch]

            pad_token_id = train_data.tokenizer.pad_token_id

            batch[0], batch[1] = trim_batch(batch[0], pad_token_id, batch[1])
            batch[2], batch[3] = trim_batch(batch[2], pad_token_id, batch[3])

            loss = model(input_ids=batch[0], attention_mask=batch[1],
                         decoder_input_ids=batch[2], decoder_attention_mask=batch[3],
                         is_training=True)
            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu.
            if torch.isnan(loss).data:
                logger.info("Stop training because loss=%s" % (loss.data))
                stop_training = True
                break
            train_losses.append(loss.detach().cpu())
            loss.backward()

            if global_step % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), args.max_grad_norm)
                optimizer.step()    # We have accumulated enough gradients
                scheduler.step()
                model.zero_grad()

            if global_step % args.eval_period == 0:
                model.eval()
                curr_performance = inference(
                    model if args.n_gpu == 1 else model.module, dev_data, args=args, save_predictions=False, logger=logger)
                logger.info("Step %d Train loss %.2f %s %s on epoch=%d" % (
                    global_step,
                    np.mean(train_losses),
                    dev_data.metric,
                    curr_performance,
                    epoch))
                train_losses = []

                def is_improved(best, curr):
                    if best is None:
                        return True
                    return any([best[m] < curr[m] for m in best])

                if is_improved(best_performance, curr_performance):
                    best_model_state_dict = {k: v.cpu() for (
                        k, v) in model.state_dict().items()}
                    # save results
                    logger.info("New best performance %s: %s -> %s on epoch=%d, global_step=%d" %
                                (dev_data.metric, best_performance, curr_performance, epoch, global_step))
                    best_model_path = os.path.join(
                        args.output_dir, "best-model.pt")
                    with open(best_model_path.replace(".pt", "_results.json"), "w") as f:
                        json.dump(curr_performance, f)
                    logger.info(
                        "Saving the new best model to {}".format(best_model_path))
                    torch.save(best_model_state_dict, best_model_path)
                    best_performance = curr_performance
                    wait_step = 0
                    stop_training = False
                else:
                    wait_step += 1
                    if wait_step >= args.wait_step:
                        stop_training = True
                        break

                model.train()

            if global_step >= args.total_steps:
                stop_training = True
                break

        if stop_training:
            break

    # model_state_dict = {k:v.cpu() for (k, v) in model.state_dict().items()}
    # torch.save(model_state_dict, os.path.join(args.output_dir, "last-model.pt"))
    return best_performance, best_model_state_dict


def inference(model, dev_data, save_predictions=False, verbose=False, args=None, logger=None, predictions_only=False):
    model.eval()
    predictions = []
    bos_token_id = dev_data.tokenizer.bos_token_id
    loss = []   # if needed
    logger.info("Starting inference ...")
    for batch in tqdm(dev_data.dataloader, desc="Inference", disable=not verbose):
        if args.n_gpu > 1:
            batch = [b.to(torch.device("cuda")) for b in batch]
        pad_token_id = dev_data.tokenizer.pad_token_id

        batch[0], batch[1] = trim_batch(batch[0], pad_token_id, batch[1])

        outputs = model.generate(input_ids=batch[0],
                                 attention_mask=batch[1],
                                 num_beams=dev_data.args.num_beams,
                                 max_length=dev_data.args.max_output_length,
                                 decoder_start_token_id=model.config.bos_token_id,
                                 early_stopping=dev_data.gen_early_stop,)

        for _, output in zip(batch[0], outputs):
            pred = dev_data.decode(output)
            predictions.append(pred)
    logger.info("Starting inference ... Done")

    if predictions_only:
        return predictions
    if save_predictions:
        dev_data.save_predictions(predictions, prefix=args.prefix)
    logger.info("Starting evaluation metric ...")
    result = dev_data.evaluate(predictions, verbose=verbose)
    logger.info("Starting evaluation metric ... Done!")
    return predictions, result, loss
