from contextlib import nullcontext
from typing import Mapping, Tuple
import dataclasses
import math

import torch
from torch.nn.parallel import DistributedDataParallel

from gptplay import utils


@torch.no_grad()
def estimate_loss(model, get_loss, data_loader, eval_iters, amp_ctx, is_cuda, device)->Tuple[float, float]:
    model.eval()
    loss_sum, correct_sum, data_count = 0., 0, 0
    for i, (x, y) in enumerate(data_loader):
        if eval_iters is not None and i >= eval_iters: # eval_iters is None means eval the whole dataset
            break
        x, y = x.pin_memory().to(device, non_blocking=True) if is_cuda else x.to(device), \
               y.pin_memory().to(device, non_blocking=True) if is_cuda else y.to(device)
        with amp_ctx:
            logits = model(x)
            loss, correct = get_loss(logits, y)
            loss_sum += loss.item() * len(y)
            correct_sum += correct.item()
            data_count += len(y)
    model.train()
    return loss_sum / data_count, correct_sum / data_count

def log_metrics(logger, step, model, get_loss, eval_iters, lr,
                amp_ctx, is_cuda, device, train_loader, val_loader, test_loader):

    train_loss, train_acc = estimate_loss(model, get_loss, train_loader, eval_iters,
                                    amp_ctx, is_cuda, device)

    val_loss, val_acc = estimate_loss(model, get_loss, val_loader, eval_iters,
                                    amp_ctx, is_cuda, device)

    w_norm = model.weight_norm()

    metrics = {
        "train/step": step,
        "train/loss": train_loss,
        "train/ppl": math.exp(train_loss),
        "train/acc": train_acc,
        "val/loss": val_loss,
        "val/ppl": math.exp(val_loss),
        "val/acc": val_acc,
        "w_norm": w_norm,
        "lr": lr,
    }

    if test_loader:
        test_loss, test_acc = estimate_loss(model, get_loss, test_loader, eval_iters,
                                    amp_ctx, is_cuda, device)
        metrics["test/loss"] = test_loss,
        metrics["test/ppl"] = math.exp(test_loss),
        metrics["test/acc"] = test_acc,

    logger.info(metrics)

    return val_loss

def clean(config:Mapping)->Mapping:
    """Remove module key from config so we can pass it as arguments to functions."""
    c = config.copy()
    c.pop('module')
    return c

def train(config:Mapping, logger):
    project_name = config['logging']['project_name']
    run_name = config['logging']['run_name']
    device_type = config['general']['device_type']
    dtype = config['general']['dtype']
    enable_distributed = config['general']['enable_distributed']
    gradient_accumulation_steps = config['training']['gradient_accumulation_steps']
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

    get_data = utils.import_fn(config['data']['module'])
    get_tokenizer = utils.import_fn(config['tokenizer']['module'])
    get_optim = utils.import_fn(config['optimizer']['module'])
    get_scheduler = utils.import_fn(config['scheduler']['module'])
    get_model = utils.import_fn(config['model']['module'])
    get_loss = utils.import_fn(config['loss']['module'])

    torch_info = utils.setup_torch(seed=seed,
                device_type=device_type, dtype=dtype,
                enable_distributed=enable_distributed,
                gradient_accumulation_steps_1gpu=gradient_accumulation_steps)

    utils.setup_sys(seed + torch_info.seed_offset)

    logger.summary(dataclasses.asdict(torch_info))
    logger.summary({"global_batch_size": gradient_accumulation_steps * train_batch_size * torch_info.world_size,
                    "local_batch_size": gradient_accumulation_steps * torch_info.world_size,
                    "tokens_per_iter": gradient_accumulation_steps * train_batch_size * torch_info.world_size * context_length
                    })

    device = torch.device(torch_info.device_name)
    amp_ctx = nullcontext() if torch_info.device_type == 'cpu' else torch.amp.autocast(device_type=torch_info.device_type, dtype=torch_info.pt_dtype)

    # get dataset
    train_loader, val_loader, test_loader = get_data(local_rank=torch_info.local_rank,
                                                     **clean(data_config))
    tokenizer = get_tokenizer(**clean(tokenizer_config))

    logger.summary({'vocab_size': len(tokenizer),
                    'train_len': len(train_loader.dataset),
                    'val_len': len(val_loader.dataset),
                    'test_len': len(test_loader.dataset) if test_loader is not None else 0,
                    'train_batches': len(train_loader),
                    'val_batches': len(val_loader),
                    'test_batches': len(test_loader) if test_loader is not None else 0
                    })

    # create model
    model = get_model(vocab_size=len(tokenizer),
                      **clean(model_config)).to(device)

    # optimizer
    optimizer = get_optim(model.named_parameters(),
                          enable_fused=torch_info.is_cuda,
                          **clean(optimizer_config))

    if torch_compile:
        logger.info("Compiling model...")
        try:
            model = torch.compile(model) # requires PyTorch 2.0
        except Exception as e:
            logger.error(f"Failed to compile model: {str(e)}")
        logger.info("Compiling done.")

    if torch_info.is_distributed:
        model = DistributedDataParallel(model,
                                        device_ids=[torch_info.local_rank])

    # scheduler provides warmup and then constant lr
    scheduler = get_scheduler(optimizer, **clean(scheduler_config))

    # initialize a GradScaler. If enabled=False scaler is a no-op
    scaler = torch.cuda.amp.GradScaler(enabled=(torch_info.pt_dtype == torch.float16))

    step, eval_count = 0, 0
    epoch, epoch_step = 0, 0 # epoch steps is useful to know how many epochs we did
    best_val_loss, evel_count = float('inf'), 0

    if torch_info.is_master:
        out_dir = utils.full_path(out_dir, create=True)
        logger.info({'out_dir': out_dir})

    # run steps
    while step < num_steps:
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
            x, y = x.pin_memory().to(device, non_blocking=True) if torch_info.is_cuda else x.to(device), \
                   y.pin_memory().to(device, non_blocking=True) if torch_info.is_cuda else y.to(device)

            loss_sum, correct_sum, data_count = 0., 0, 0
            for micro_step in range(gradient_accumulation_steps):
                if torch_info.is_distributed:
                    # Instead of model.no_sync(), we do Karpathy's hack
                    # On last step, flag model to require backward grad sync
                    model.require_backward_grad_sync = (micro_step == gradient_accumulation_steps - 1)
                with amp_ctx:
                    logits = model(x)
                    loss, correct = get_loss(logits, y)

                    loss_sum += loss.item() * len(y)
                    correct_sum += correct.item()
                    data_count += len(y)

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
                    "train/step_acc": correct_sum / data_count,
                    "train/step_loss": loss_sum / data_count,
                    "train/step_ppl": math.exp(loss_sum / data_count),
                }
                logger.info(metrics)

            if torch_info.is_master and (step+1) % eval_every == 0 or step+1 >= num_steps:
                eval_count += 1
                val_loss = log_metrics(logger, step, model, get_loss, eval_iters,
                    optimizer.param_groups[0]['lr'],
                    amp_ctx, torch_info.is_cuda, device, train_loader, val_loader,
                    test_loader if step+1 >= num_steps else None)

                if val_loss < best_val_loss and \
                        ((step+1 >= num_steps) or \
                            (step > checkoint_after and eval_count % checkpoint_every == 0)):
                    best_val_loss = val_loss
                    utils.save_checkpoint(out_dir, f'{project_name}_{run_name}' ,
                                          model.module if torch_info.is_distributed else model,
                                          optimizer, scheduler, step, best_val_loss)

            step += 1
            epoch_step += 1
            if step >= num_steps:
                break

        epoch += 1

    if torch_info.is_distributed:
        torch.distributed.distroy_process_group()
