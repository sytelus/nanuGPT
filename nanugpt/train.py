from typing import Mapping, Tuple, Optional, Dict, List, Callable, MutableMapping, Mapping
import os
import timeit
import math

import torch
from torch.nn.parallel import DistributedDataParallel
from torch import distributed as dist

from nanugpt import utils
from nanugpt.config import Config
from nanugpt import common

@torch.no_grad()
def estimate_loss(model:torch.nn.Module, get_loss:Callable,
                  data_loader, eval_iters:Optional[int],
                  amp_ctx, is_cuda:bool, device)->Tuple[float, float]:
    model.eval()
    loss_sum, correct_sum, preds_count, sample_count = 0., 0, 0, 0
    for i, (x, y) in enumerate(data_loader):
        if eval_iters is not None and i >= eval_iters: # eval_iters is None means eval the whole dataset
            break
        x, y = x.pin_memory().to(device, non_blocking=True) if is_cuda else x.to(device), \
               y.pin_memory().to(device, non_blocking=True) if is_cuda else y.to(device)
        with amp_ctx:
            logits = model(x)
            loss, correct, n_preds = get_loss(logits, y)
            n_samples = len(y)
            loss_sum += loss.item() * n_samples # loss is average so we need to multiply by n_samples to get total loss over batch
            correct_sum += correct.item()
            preds_count += n_preds
            sample_count += n_samples
    model.train()
    return loss_sum / sample_count, correct_sum / preds_count


def train(config:Mapping, logger=None):
    gradient_accumulation_steps = config['training']['gradient_accumulation_steps']
    adj_grad_acc_gpu_count = config['training']['adj_grad_acc_gpu_count']
    project_name = config['logging']['project_name']
    run_name = config['logging']['run_name']
    train_batch_size = config['training']['train_batch_size']
    num_steps = config['training']['num_steps']
    grad_clip = config['training']['grad_clip']
    enable_train_log = config['training']['enable_train_log']
    train_log_every = config['training']['log_every']
    eval_every = config['eval']['eval_every']
    eval_iters = config['eval']['eval_iters']
    save_checkpoint = config['eval']['save_checkpoint']
    checkpoint_every_eval = config['eval']['checkpoint_every_eval']
    checkpoint_keep_best = config['eval']['checkpoint_keep_best']

    checkoint_after = config['eval']['checkoint_after']
    out_dir = config['general']['out_dir']
    data_config = config['data']
    context_length = config['model']['module_kwargs']['context_length'] #TODO: refactor so trainer is independent of context length?
    optimizer_config = config['optimizer']
    scheduler_config = config['scheduler']
    loss_config = config['loss']

    get_data = utils.import_fn(data_config['module'])
    get_optim = utils.import_fn(optimizer_config['module'])
    get_scheduler = utils.import_fn(scheduler_config['module'])
    get_loss = utils.import_fn(loss_config['module'])

    # setup system, device, logger, torch
    own_logger = logger is None
    device, amp_ctx, logger, torch_info = common.setup_device(config, logger)

    # adjust gradient accumulation steps if we are doing distributed training
    if torch_info.is_distributed and adj_grad_acc_gpu_count:
        assert gradient_accumulation_steps % torch_info.world_size == 0, f'gradient_accumulation_steps ({gradient_accumulation_steps}) must be divisible by ddp_world_size ({torch_info.world_size})'
        gradient_accumulation_steps = gradient_accumulation_steps // torch_info.world_size

    logger.summary({"global_batch_size": gradient_accumulation_steps * train_batch_size * torch_info.world_size,
                    "local_batch_size": gradient_accumulation_steps * torch_info.world_size,
                    "tokens_per_iter": gradient_accumulation_steps * train_batch_size * torch_info.world_size * context_length
                    })

    # create model and tokenizer
    model, tokenizer, model_config, tokenizer_config = common.create_model_tokenizer(config, logger, device)

    # get dataset
    train_loader, val_loader, test_loader = get_data(local_rank=torch_info.local_rank,
                                                     **data_config['module_kwargs'])
    logger.summary({'vocab_size': len(tokenizer),
                    'train_dataset_len': len(train_loader.dataset),
                    'val_dataset_len': len(val_loader.dataset),
                    'test_dataset_len': len(test_loader.dataset) if test_loader is not None else 0,
                    'train_dataloader_len': len(train_loader),
                    'val_dataloader_len': len(val_loader),
                    'test_dataloader_len': len(test_loader) if test_loader is not None else 0
                    })

    # optimizer
    optimizer = get_optim(model,
                          enable_fused=torch_info.is_cuda,
                          **optimizer_config['module_kwargs'])

    if torch_info.is_distributed:
        model = DistributedDataParallel(model,
                                        device_ids=[torch_info.local_rank])

    # scheduler provides warmup and then constant lr
    scheduler = get_scheduler(optimizer, **scheduler_config['module_kwargs'])

    # initialize a GradScaler. If enabled=False scaler is a no-op
    # we need loss scaling only for fp16 due to reduced precision, not bf16 or fp32
    scaler = torch.cuda.amp.GradScaler(enabled=(torch_info.pt_dtype == torch.float16))

    if torch_info.is_master:
        out_dir = utils.full_path(out_dir, create=True)
        logger.summary({'out_dir': out_dir})

    step, eval_count, iters_since_eval, total_samples, token_count = 0, 0, 0, 0, 0
    epoch, epoch_step = 0, 0 # epoch steps is useful to know how many epochs we did
    best_train_loss, best_val_loss = float('inf'), float('inf')
    best_train_loss_step, best_val_loss_step, last_checkpoint_step = -1, -1, -1
    checkpoint_log = []
    loop_start_time = last_eval_time = timeit.default_timer()

    # run steps
    while step < num_steps:
        step_start_time = timeit.default_timer()
        epoch_step = 0 # step within the epoch

        batch_iter = iter(train_loader) # restart iterator
        try:
            x, y = next(batch_iter)
            batch_iter_done = False
        except StopIteration:
            break # empty dataset

        # Loop over the training set
        while not batch_iter_done:
            model.train()
            loss_sum, correct_sum, step_preds_count = 0., 0, 0
            metrics = {} # add metrics here if any
            step_sample_count = 0

            for micro_step in range(gradient_accumulation_steps):
                x, y = x.pin_memory().to(device, non_blocking=True) if torch_info.is_cuda else x.to(device), \
                    y.pin_memory().to(device, non_blocking=True) if torch_info.is_cuda else y.to(device)

                if torch_info.is_distributed:
                    # Instead of model.no_sync(), we do Karpathy's hack
                    # On last step, flag model to require backward grad sync
                    model.require_backward_grad_sync = (micro_step == gradient_accumulation_steps - 1)

                # note that we don't take context length into account here
                # if we have N tokens in dataset, we move window N times and do
                # N iterations per epoch. So, token count in traditional sense is
                # number of samples, not number of tokens.
                n_samples = len(x)
                step_sample_count += n_samples
                token_count += x.numel()

                with amp_ctx:
                    logits = model(x)
                    loss, correct, n_preds = get_loss(logits, y)

                    loss_sum += loss.item() * n_samples
                    correct_sum += correct.item()
                    step_preds_count += n_preds

                    # Scale the loss to account for gradient accumulation
                    # During gradient accumulation, gradients are summed at each micro step.
                    # When we divide the loss by the number of micro steps we average out the gradients
                    # so that the net value of grads is same as if we had a larger batch size
                    loss = loss / gradient_accumulation_steps # scale the loss to account for gradient accumulation

                # immediately async prefetch next batch while model is doing the forward pass on the GPU
                try:
                    x, y = next(batch_iter)
                except StopIteration:
                    batch_iter_done = True

                # backward pass, with gradient scaling if training in fp16
                scaler.scale(loss).backward()

                # if we run out of data, break out of the micro steps loop
                # this means last batch is partial
                if batch_iter_done:
                    break

            # --- end of gradient accumulation loop ---

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

            iters_since_eval += 1

            if torch_info.is_distributed:
                # reduce tensors to rank 0 to get numbers from all ranks
                loss_sum_dist = torch.tensor(loss_sum, dtype=torch.float32, device=device)
                counts_dist = torch.tensor([correct_sum, step_preds_count, step_sample_count], dtype=torch.int64, device=device)
                dist.reduce(loss_sum_dist, dst=0, op=dist.ReduceOp.SUM)
                dist.reduce(counts_dist, dst=0,op=dist.ReduceOp.SUM)
                if torch_info.is_master:
                    loss_sum = loss_sum_dist.item()
                    correct_sum, step_preds_count, step_sample_count = tuple(counts_dist.tolist())

            total_samples += step_sample_count
            train_loss = loss_sum / step_sample_count
            train_acc = correct_sum / step_preds_count
            if train_loss < best_train_loss:
                best_train_loss = train_loss
                best_train_loss_step = step

            if enable_train_log and torch_info.is_master and (step % train_log_every == 0 or step+1 >= num_steps):
                metrics.update({
                    "train/step": step,
                    "train/loss": train_loss,
                    "train/acc": train_acc,
                    "train/ppl": math.exp(train_loss),
                    "train/best_loss": best_train_loss,
                    "train/best_loss_step": best_train_loss_step,
                    "train/epoch": epoch,
                    "train/epoch_step": epoch_step,
                    "train/step_interval": timeit.default_timer() - step_start_time,
                    "train/samples": total_samples,
                    "train/step_samples": step_sample_count,
                    "train/token_count": token_count,
                    "train/tokens_per_sec": token_count / (timeit.default_timer() - loop_start_time),
                    "lr": optimizer.param_groups[0]['lr'],
                    "elapsed_s": timeit.default_timer() - loop_start_time
                })

            # is it time to evaluate? We evaluate after 1st step to get initial loss.
            if torch_info.is_master and (step % eval_every == 0 or step+1 >= num_steps):
                eval_count += 1
                eval_interval = timeit.default_timer() - last_eval_time

                model_kwargs = model_config['module_kwargs']
                transformer_tflops = utils.transformer_tflops(batch_size=train_batch_size,
                    param_count=utils.module_params_count(model, non_embedding=True),
                    context_length=context_length, dt=eval_interval, iterations=iters_since_eval,
                    n_embd=model_kwargs['n_embd'], n_layer=model_kwargs['n_layer'], n_head=model_kwargs['n_head'])

                val_loss, val_acc = estimate_loss(model, get_loss, val_loader, eval_iters,
                                                amp_ctx, torch_info.is_cuda, device)
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_val_loss_step = step

                metrics.update({
                    "val/loss": val_loss,
                    "val/acc": val_acc,
                    "val/ppl": math.exp(val_loss),
                    "val/best_loss": best_val_loss,
                    "val/best_loss_step": best_val_loss_step,
                    "w_norm": utils.weight_norm(model),
                    "val/interval": eval_interval,
                    "transformer_tflops": transformer_tflops
                })
                if step+1 >= num_steps and test_loader:
                    test_loss, test_acc = estimate_loss(model, get_loss, test_loader, None,
                                                amp_ctx, torch_info.is_cuda, device)
                    metrics.update({
                        "test/loss": test_loss,
                        "test/ppl": math.exp(test_loss),
                        "test/acc": test_acc,
                    })
                last_eval_time, iters_since_eval = timeit.default_timer(), 0

                # if this is the best model so far, save it
                if save_checkpoint and last_checkpoint_step < best_val_loss_step and \
                        ((step+1 >= num_steps) or \
                            (step > checkoint_after and (eval_count+1) % checkpoint_every_eval == 0)
                        ):
                    checkpoint_filename = project_name + \
                        (f"_{run_name}" if run_name else "") + \
                        f"_{step}" if not checkpoint_keep_best else "_best"
                    checkpoint_filepath = utils.save_checkpoint(out_dir, checkpoint_filename,
                                          model.module if torch_info.is_distributed else model,
                                          optimizer, scheduler, step, best_val_loss)

                    metrics.update({"checkpoint_filepath": checkpoint_filepath})

                    checkpoint_log.append(metrics)

            if len(metrics) > 0:
                logger.info(metrics)

            step += 1
            epoch_step += 1
            if step >= num_steps:
                break

        epoch += 1


    if torch_info.is_master:
        checkpoint_log_filepath = os.path.join(out_dir, "checkpoint_log.yaml")
        utils.save_yaml(checkpoint_log, checkpoint_log_filepath)
        logger.log_artifact('checkpoint_log', 'file', file_or_dir=checkpoint_log_filepath)

        if torch_info.is_distributed:
            dist.destroy_process_group()

        if own_logger:
            logger.all_done()


if __name__ == "__main__":
    # specify config file to use as first argument in commandline
    config = Config(default_config_filepath='configs/grokking/prime223.yaml')
    train(config)