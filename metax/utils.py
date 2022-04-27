from metax.config import METRICS
import wandb
import torch
import os
from transformers import AdamW, get_linear_schedule_with_warmup
from transformers.optimization import get_constant_schedule
from logging import WARN
from metax.models.utils import trim_batch

import copy
import json
import logging
import os
import random
import time
import numpy as np
from datetime import datetime
from logging import INFO, WARN


def last_n_increasing(l, n=3, thres=1e-2):
    if len(l) <= n:
        return False
    # l is for loss
    for i in range(len(l) - n, len(l)):
        if l[i - 1] < l[i] - thres:
            return True
    return False


def get_optimizer(model,
                  learning_rate=3e-5,
                  weight_decay=0.01,
                  adam_epsilon=1e-8,
                  warmup_steps=0,
                  total_steps=-1):
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [{
        'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
        'weight_decay': weight_decay
    }, {
        'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
        'weight_decay': 0.0
    }]
    optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate, eps=adam_epsilon)
    if warmup_steps <= 0:
        scheduler = get_constant_schedule(optimizer)
    else:
        assert total_steps > 0
        scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=warmup_steps,
                                                num_training_steps=total_steps)
    return optimizer, scheduler


def model_train(
    model,
    optimizer,
    dataloader,
    run_name='train',
    gradient_steps=1,
    max_grad_norm=0.1,
    num_epochs=200,
    gpus=[0, 1],
    scheduler=None,
    logger=None,
    evaluate_dataloader=None,
    loss_evaluation=False,
    evaluation_steps=100,
    early_stop=False,
    save_model=False,
    saving_steps=100,
    total_steps=-1,
    saving_dir=None,
    unfreezed_layers=1,):
    """
    Standardized training procedure

    """
    if logger:
        logger.log(WARN, 'Start training!')
    
    assert unfreezed_layers >= 0

    if unfreezed_layers == 0:
        logger.info('training with no layers freezed')
        model.train()
    else:
        logger.info(f'training with last {unfreezed_layers} layers of encoder & decoder unfreezed ')
        model.train()
        for i, m in enumerate(model.module.encoder.block):
            #Only un-freeze the last n transformer blocks in the decoder
            if i < len(model.module.encoder.block) - unfreezed_layers:
                for parameter in m.parameters():
                    parameter.requires_grad = False 
        for i, m in enumerate(model.module.decoder.block):        
            #Only un-freeze the last n transformer blocks in the decoder
            if i < len(model.module.decoder.block) - unfreezed_layers:
                for parameter in m.parameters():
                    parameter.requires_grad = False 

    global_step = 0
    update_step = 0
    dev_losses = []
    stop_training = False
    print(f"int(num_epochs)* len(dataloader.dataloader) * gradient_steps={int(num_epochs)* len(dataloader.dataloader)* gradient_steps}")
    print(f"total_steps={total_steps}")
    min_dev_loss = 1e8
    no_decrease = 0
    best_model = None
    for epoch in range(int(num_epochs)):
        for idx, batch in enumerate(dataloader.dataloader):
            # logger.info(f"\tTraining. epoch #{epoch+1} / {num_epochs}: batch #{idx}/{total_steps}")
            logger.info(f"\tTraining. step #{update_step+1} / {total_steps}") # TODO: this is not considering accumulation
            # here the batch is a mini-batch of the current data batch
            # use train dataloader
            if gpus:
                batch = [b.to(torch.device("cuda")) for b in batch]

            pad_token_id = dataloader.tokenizer.pad_token_id
            batch[0], batch[1] = trim_batch(batch[0], pad_token_id, batch[1])
            batch[2], batch[3] = trim_batch(batch[2], pad_token_id, batch[3])
            loss = model(input_ids=batch[0],
                         attention_mask=batch[1],
                         decoder_input_ids=batch[2],
                         decoder_attention_mask=batch[3],
                         is_training=True)

            if gpus and len(gpus) > 1:
                loss = loss.mean()  # mean() to average on multi-gpu.

            def get_lr():
                for group in optimizer.param_groups:
                    return group['lr']

            loss.backward()

            global_step += 1
            if global_step % gradient_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()  # We have accumulated enough gradients
                update_step += 1
                if scheduler:
                    scheduler.step()
                model.zero_grad()
                

            
            if loss_evaluation and update_step % evaluation_steps == 0:
                dev_loss = loss_evaluate(model, evaluate_dataloader)
                model.train()
                wandb.log({'loss': loss, 'learning rate': get_lr(), 'validation loss': dev_loss}) 
                dev_losses.append(dev_loss)
                
                if early_stop:
                    if dev_loss <= min_dev_loss:
                        min_dev_loss = dev_loss
                        no_decrease = 0
                        best_model = None
                        best_model = copy.deepcopy(best_model)
                        logger.info(f'Found a new best model @ {update_step}')
                    else:
                        no_decrease += 1
                    if no_decrease >=20: # TODO: patience
                        if logger:
                            logger.log(WARN,
                                    f'Stopped training early because of growing evaluation losses. trained for {epoch} epochs')
                            logger.log(WARN,
                                    f'{dev_losses}')
                        stop_training = True
                        break
            else:
                wandb.log({'loss': loss, 'learning rate': get_lr()})

            if save_model and (global_step > 1 and update_step % saving_steps==0):
                path_to_save_model = os.path.join(saving_dir, f'{run_name}-{update_step}.pt')
                torch.save(
                    {
                        'min_dev_loss': dev_losses[-1] if len(dev_losses) > 0 else -1,
                        'global_step': update_step,
                        'optimizer_state': optimizer.state_dict(),
                        'model_state': {k: v.cpu() for k, v in model.state_dict().items()},
                    }, path_to_save_model)
                logger.info(f"Saved the model to {path_to_save_model}!")
            if update_step >= total_steps:
                logger.info(f"update_step ({update_step}) == total_steps ({total_steps}) --> Stop Training")
                stop_training = True
                break          
              
        if stop_training:
            break

    if best_model:
        return best_model
    return model
    


def metric_evaluate(
    model_in,
    dataloader,
    save_predictions=False,
    args=None,
    logger=None,
    predictions_only=False,
    prefix='', ):
    model_in.eval()
    model = model_in if args.n_gpu == 1 else model_in.module
    predictions = []
    bos_token_id = dataloader.tokenizer.bos_token_id
    logger.info(f"Starting inference for {args.run_name} - {prefix} ...")
    for idx, batch in enumerate(dataloader.dataloader):
        if idx % 10 == 0:
            logger.info(f"Current Progress for Inference (in batches): {idx}/{len(dataloader.dataloader)}")
        if args.n_gpu > 0:
            batch = [b.to(torch.device("cuda")) for b in batch]
        pad_token_id = dataloader.tokenizer.pad_token_id

        batch[0], batch[1] = trim_batch(batch[0], pad_token_id, batch[1])

        outputs = model.generate(
            input_ids=batch[0],
            attention_mask=batch[1],
            num_beams=dataloader.args.num_beams,
            max_length=dataloader.args.max_output_length,
            decoder_start_token_id=model.config.bos_token_id,
            early_stopping=dataloader.gen_early_stop,
        )
        # embed()

        for _, output in zip(batch[0], outputs):
            pred = dataloader.decode(output)
            # TODO: post-process the prediction output.
            # 1. strip()
            # 2. remove duplicate  
            pred = pred.strip()
            words = pred.split()
            if len(words) == 2 and words[0] == words[1]:
                new_pred = " ".join(words[1:])
                logger.info(f"post-processing '{pred}' --> '{new_pred}'")
                pred = new_pred
            predictions.append(pred)
    logger.info("Starting inference ... Done")

    if predictions_only:
        return predictions
    if save_predictions:
        dataloader.save_predictions(predictions, prefix=str(prefix))
    logger.info("Starting evaluation metric ...")
    result = dataloader.evaluate(predictions)
    logger.info("Starting evaluation metric ... Done!")
    return predictions, result


def loss_evaluate(model, dataloader, gpus=[0, 1]):
    model.eval()
    losses = []
    for batch in dataloader.dataloader:
        if gpus:
            batch = [b.to(torch.device("cuda")) for b in batch]

        pad_token_id = dataloader.tokenizer.pad_token_id
        batch[0], batch[1] = trim_batch(batch[0], pad_token_id, batch[1])
        batch[2], batch[3] = trim_batch(batch[2], pad_token_id, batch[3])
        # TODO fix bias of the avg
        loss = model(input_ids=batch[0],
                     attention_mask=batch[1],
                     decoder_input_ids=batch[2],
                     decoder_attention_mask=batch[3],
                     is_training=True)
        if gpus and len(gpus) > 1:
            loss = loss.mean()
        losses.append(loss.item())
    return torch.tensor(losses).mean().item()
