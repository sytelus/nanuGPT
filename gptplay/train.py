from math import ceil
import time
from typing import Mapping, Tuple
import os
import dataclasses

import numpy as np

import torch
from torch.nn.parallel import DistributedDataParallel

from gptplay.datasets.tokenized_data import get_data
from gptplay.tokenizers.tiktoken import get_tokenzier
from gptplay.optimizers.adam_w import get_optim
from gptplay.schedulers.nanogpt_cosine import get_scheduler
from gptplay.models.tiny_transformer import get_model
from gptplay import utils

@torch.no_grad()
def estimate_loss(model, criterion, data_loader, eval_iters, amp_ctx, is_cuda, device):
    model.eval()
    loss_sum = 0.
    count = 0
    for i, x, y in enumerate(data_loader)):
        if i >= eval_iters:
            break
        x, y = x.pin_memory().to(device, non_blocking=True) if is_cuda else x.to(device), \
               y.pin_memory().to(device, non_blocking=True) if is_cuda else y.to(device)

        with amp_ctx:
            logits = model(x, y)
            loss_sum += criterion(logits.view(-1, logits.size(-1)), targets.view(-1),
                                  ignore_index=-1).item() * len(x)
            correct += (torch.argmax(logits[-1,:,:], dim=1) == y).sum().item()
            count += len(x)
    model.train()
    return loss_sum / count, correct / count

def log_metrics(logger, step, model, criterion, eval_iters,
                amp_ctx, is_cuda, device, train_loader, val_loader, test_loader):

    train_loss, train_acc = estimate_loss(model, criterion, train_loader, eval_iters,
                                    amp_ctx, is_cuda, device)

    val_loss, val_acc = estimate_loss(model, criterion, val_loader, eval_iters,
                                    amp_ctx, is_cuda, device)

    w_norm = model.weight_norm()

    metrics = {
        "train/step": step,
        "train/loss": train_loss.item(),
        "train/ppl": math.exp(train_loss.item()),
        "train/acc": train_acc.item(),
        "val/loss": val_loss.item(),
        "val/ppl": math.exp(val_loss.item()),
        "val/acc": val_acc.item(),
        "w_norm": w_norm,
        "lr": optimizer.param_groups[0]['lr'],
    }

    if test_loader:
        test_loss, test_acc = estimate_loss(model, criterion, test_loader, eval_iters,
                                    amp_ctx, is_cuda, device)
        metrics["test/loss"] = test_loss.item(),
        metrics["test/ppl"] = math.exp(test_loss.item()),
        metrics["test/acc"] = test_acc.item(),

    logger.info(metrics)

    return val_loss

def train(config:Mapping, logger):
    project_name = config['general']['project_name']
    run_name = config['general']['run_name']
    device_name = config['general']['device']
    dtype = config['general']['dtype']
    enable_distributed = training_config['enable_distributed']
    gradient_accumulation_steps = training_config['gradient_accumulation_steps']
    train_batch_size = config['data']['train_batch_size']
    seed = config['general']['seed']
    torch_compile = config['general']['torch_compile']
    num_steps = config['training']['num_steps']
    grad_clip = config['training']['grad_clip']
    train_log_every = config['training']['log_every']
    eval_every = config['eval']['eval_every']
    eval_iters = config['eval']['eval_iters']
    checkpoint_every = config['eval']['checkpoint_every']
    checkoint_after = config['eval']['checkoint_after']
    out_dir = config['general']['out_dir']
    data_config = config['data']
    model_config = config['model']
    context_length = config['model']['context_length']
    optimizer_config = config['optimizer']
    scheduler_config = config['scheduler']
    tokenizer_config = config['tokenizer']

    torch_info = utils.setup_torch(seed=seed,
                device_name=device_name,
                enable_distributed=enable_distributed,
                gradient_accumulation_steps_1gpu=gradient_accumulation_steps)

    utils.setup_sys(seed + torch_info.seed_offset)

    logger.summary(dataclasses.asdict(torch_info))
    logger.summary({"global_batch_size": gradient_accumulation_steps * train_batch_size * torch_info.world_size,
                    "local_batch_size": gradient_accumulation_steps * torch_info.world_size,
                    "tokens_per_iter": gradient_accumulation_steps * train_batch_size * torch_info.world_size * context_length
                    })

    device = torch.device(torch_info.device_name)
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
    amp_ctx = nullcontext() if torch_info.device_name == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

    # get dataset
    train_loader, val_loader, test_loader = get_data(local_rank=torch_info.local_rank,
                                                     **data_config)
    tokenizer = get_tokenzier(**tokenizer_config)

    logger.summary({'vocab_size': len(tokenizer),
                    'train_len': len(train_loader.dataset),
                    'val_len': len(val_loader.dataset),
                    'test_len': len(test_loader.dataset) if test_loader is not None else 0,
                    'train_batches': len(train_loader),
                    'val_batches': len(val_loader),
                    'test_batches': len(test_loader) if test_loader is not None else 0
                    })

    # create model
    model = get_model(vocab_size=len(tokenizer), **model_config).to(device)

    # optimizer
    optimizer = get_optim(model.parameters(),
                          enable_fused=torch_info.is_cuda,
                          **optimizer_config)

    if torch_compile:
        logger.info("Compiling model...")
        model = torch.compile(model) # requires PyTorch 2.0
        logger.info("Compiling done.")

    if torch_info.is_distributed:
        model = DistributedDataParallel(model,
                                        device_ids=[torch_info.local_rank],
                                        output_device=torch_info.local_rank)

    # scheduler provides warmup and then constant lr
    scheduler = get_scheduler(optimizer, **scheduler_config)

    step = 0
    epoch, epoch_step = 0, 0 # epoch steps is useful to know how many epochs we did
    best_val_loss, evel_count = float('inf'), 0

    criterion = torch.nn.CrossEntropyLoss()
    model.train()

    if torch_info.is_master:
        out_dir = utils.full_path(out_dir, create=True)
        logger.info({'out_dir': out_dir})

    batch_iter = iter(train_loader)
    try:
        batch = next(batch_iter)
        batch_iter_done = False
    except StopIteration:
        batch_iter_done = True
        batch_iter = iter(train_loader)

    # run steps
    while step < num_steps:
        epoch_step = 0
        # Loop over the training set
        while not batch_iter_done:
            model.train()
            x, y = tuple(t.pin_memory().to(device, non_blocking=True) \
                        if torch_info.is_cuda else t.to(device) for t in batch)

            loss_sum, acc_sum, data_count = 0., 0, 0
            for micro_step in range(gradient_accumulation_steps):
                if ddp:
                    # Instead of model.no_sync(), we do Karpathy's hack
                    # On last step, flag model to require backward grad sync
                    model.require_backward_grad_sync = (micro_step == gradient_accumulation_steps - 1)
                with amp_ctx:
                    logits = model(x, y)
                    loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1),
                                  ignore_index=-1)
                    loss = loss / gradient_accumulation_steps # scale the loss to account for gradient accumulation
                    loss_sum += loss.item() * len(y)
                    acc_sum = (torch.argmax(logits[-1,:,:], dim=1) == y).sum().item()
                    data_count += len(y)

                # immediately async prefetch next batch while model is doing the forward pass on the GPU
                try:
                    batch = next(batch_iter)
                    batch_iter_done = False
                except StopIteration:
                    batch_iter_done = True
                    batch_iter = iter(train_loader)

                # backward pass, with gradient scaling if training in fp16
                scaler.scale(loss).backward()

            # clip the gradient
            if grad_clip != 0.0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

            # step the optimizer and scaler if training in fp16
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            # flush the gradients as soon as we can, no need for this memory anymore
            optimizer.zero_grad(set_to_none=True)

            if torch_info.is_master and step % train_log_every == 0 or step+1 >= num_steps:
                metrics = {
                    "train/step": step,
                    "train/step_acc": acc_sum / data_count,
                    "train/step_loss": loss_sum / data_count,
                    "train/step_ppl": math.exp(loss_sum / data_count),
                }
                logger.info(metrics)

            if torch_info.is_master and (step+1) % eval_every == 0 or step+1 >= num_steps:
                val_loss = log_metrics(logger, step, model, criterion, eval_iters,
                    amp_ctx, torch_info.is_cuda, device, train_loader, val_loader,
                    test_loader if step+1 >= num_steps else None)

                if val_loss < best_val_loss and step > checkoint_after and eval_count % checkpoint_every == 0:
                    best_val_loss = val_loss
                    utils.save_checkpoint(out_dir, f'{project_name}_{run_name}' ,
                                          model, optimizer, scheduler,
                                          step, best_val_loss)

            step += 1
            epoch_step += 1
            if step >= num_steps:
                break
        epoch += 1

    if torch_info.is_distributed:
        torch.distributed.distroy_process_group()
