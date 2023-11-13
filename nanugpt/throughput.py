from contextlib import AbstractContextManager
from typing import Mapping, Tuple, Optional, Callable, Mapping, List
import os
import timeit
import math

import torch
from torch.nn.parallel import DistributedDataParallel
from torch import distributed as dist

from nanugpt import utils
from nanugpt import common
from nanugpt import glogging as logging
from nanugpt.config import Config


def measure_throuput(config:Mapping,
                     model_sizes:List[dict],
                     context_lengths:List[int],
                     batch_sizes:List[int],
                     logger:Optional[logging.Logger]=None):
    train_batch_size = config['training']['train_batch_size']
    out_dir = config['general']['out_dir']
    data_config = config['data']
    gradient_accumulation_steps = config['training']['gradient_accumulation_steps']
    adj_grad_acc_gpu_count = config['training']['adj_grad_acc_gpu_count']

    get_data = utils.import_fn(data_config['module'])

    # setup system, device, logger, torch
    own_logger = logger is None
    logger = common.setup_logger(utils.is_master_process(), config, logger)

    device, amp_ctx, torch_info = common.setup_device(config, logger)
    assert torch_info.is_master == utils.is_master_process(), "torch_info.is_master != utils.is_master_process()"

    # adjust gradient accumulation steps if we are doing distributed training
    if torch_info.is_distributed and adj_grad_acc_gpu_count:
        assert gradient_accumulation_steps % torch_info.world_size == 0, f'gradient_accumulation_steps ({gradient_accumulation_steps}) must be divisible by ddp_world_size ({torch_info.world_size})'
        gradient_accumulation_steps = gradient_accumulation_steps // torch_info.world_size

    if torch_info.is_master:
        out_dir = utils.full_path(out_dir, create=True)
        logger.summary({'out_dir': out_dir})

    # create tokenizer
    tokenizer, tokenizer_config = common.create_tokenizer(config, logger)
    logger.summary({'vocab_size': len(tokenizer)})

    # turn off loggig except for messages we want
    logging.get_logger().quite('params_m(c)')

    for batch_size in batch_sizes:
        for context_length in context_lengths:
            data_config['module_kwargs']['train_batch_size'] = batch_size
            data_config['module_kwargs']['context_length'] = context_length
            train_loader, val_loader, test_loader = get_data(local_rank=torch_info.local_rank,
                                                            **data_config['module_kwargs'])
            for model_size in model_sizes:
                model_kwargs = config['model']['module_kwargs']
                model_kwargs['context_length'] = context_length
                model_kwargs['n_layer'] = model_size['n_layer']
                model_kwargs['n_embd'] = model_size['n_embd']
                model_kwargs['n_head'] = model_size['n_head']

                try:
                    torch.cuda.empty_cache()

                    model, model_config, train_loss, train_acc, total_samples, token_count, loop_start_time, loop_end_time, num_steps = \
                        train_config(config, logger, device, len(tokenizer), torch_info, train_loader, amp_ctx, gradient_accumulation_steps)

                    params_nonembedding_trainable = utils.module_params_count(model)[-1]

                    model_kwargs = model_config['module_kwargs']
                    dt = loop_end_time - loop_start_time
                    transformer_tflops = utils.transformer_tflops(batch_size=train_batch_size,
                        params_nonembedding_trainable=params_nonembedding_trainable,
                        context_length=context_length, dt=dt,
                        n_embd=model_kwargs['n_embd'], n_layer=model_kwargs['n_layer'],
                        forward_iters=gradient_accumulation_steps, backward_iters=1
                    )
                    samples_rate = total_samples / dt
                    tokens_rate = token_count / dt
                    step_time = dt / num_steps
                except Exception as e:
                    logger.summary({
                        'params_m(c)': model_size['params_m'],
                        'params_m(a)': '***OOM***' if isinstance(e, RuntimeError) and 'CUDA out of memory' in str(e) else type(e).__name__,
                        'context_length': context_length,
                        'n_layer': model_size['n_layer'],
                        'n_embd': model_size['n_embd'],
                        'n_head': model_size['n_head'],
                    })
                    break

                logger.summary({
                                'params_m(c)': model_size['params_m'],
                                'params_m(a)': int(params_nonembedding_trainable/1e6),
                                'context_length': context_length,
                                'n_layer': model_size['n_layer'],
                                'n_embd': model_size['n_embd'],
                                'n_head': model_size['n_head'],
                                'samples/s': samples_rate,
                                'tokens/s': tokens_rate,
                                'step_time': step_time,
                                'transformer_tflops': transformer_tflops,
                                "gpu_batch_size": batch_size,
                                "global_batch_size": gradient_accumulation_steps * batch_size * torch_info.world_size,
                                "local_batch_size": gradient_accumulation_steps * batch_size,
                                "tokens/step": gradient_accumulation_steps * batch_size * torch_info.world_size * context_length,
                                })

    if torch_info.is_master:
        if torch_info.is_distributed:
            dist.destroy_process_group()

        if own_logger:
            logger.all_done()


def train_config(config:Mapping, logger:logging.Logger, device:torch.device,
                 vocab_size:int, torch_info:utils.TorchInfo,
                 train_loader:torch.utils.data.DataLoader, amp_ctx:AbstractContextManager,
                 gradient_accumulation_steps:int):

    num_steps = config['training']['num_steps']
    grad_clip = config['training']['grad_clip']
    optimizer_config = config['optimizer']
    scheduler_config = config['scheduler']
    loss_config = config['loss']

    get_optim = utils.import_fn(optimizer_config['module'])
    get_scheduler = utils.import_fn(scheduler_config['module'])
    get_loss = utils.import_fn(loss_config['module'])

    # create model
    model, model_config = common.create_model(config, logger, device, vocab_size=vocab_size)

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

    step, total_samples, token_count = 0, 0, 0
    loop_start_time = timeit.default_timer()
    train_loss, train_acc = 0., 0.

    # run steps
    while step < num_steps:
        step_start_time = timeit.default_timer()

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
                scaler.scale(loss).backward() # type: ignore

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

            step += 1
            if step >= num_steps:
                break

    loop_end_time = timeit.default_timer()

    return model, model_config, train_loss, train_acc, total_samples, token_count, loop_start_time, loop_end_time, num_steps

if __name__ == "__main__":
    # specify config file to use as first argument in commandline
    config = Config(default_config_filepath='configs/train_gpt2/tinystories.yaml')
    config['training']['num_steps'] = 5
    config['training']['gradient_accumulation_steps'] = 1
    config['training']['adj_grad_acc_gpu_count'] = False
    config['training']['enable_train_log'] = False
    config['logging']['log_filename'] = 'throughput.log'
    config['logging']['enable_wandb'] = False
    config['logging']['project_name'] = 'gpt2-throughput'
    config['logging']['run_name'] = 'throughput'

    model_sizes = list(common.get_model_sizes().values())
    model_sizes.sort(key=lambda x: x['params_m'])

    measure_throuput(config,
                     model_sizes=model_sizes,
                     context_lengths=[128, 256, 512, 1024, 2048, 4096, 8192, 16384],
                     batch_sizes=[1, 2, 4, 8, 12, 16, 24, 26, 32, 48, 60, 64, 128, 256, 512, 1024])
