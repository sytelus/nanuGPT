import os
from contextlib import AbstractContextManager
from typing import Iterator, Mapping, Tuple, Optional, Callable, Mapping, List

import torch
from torch.nn.parallel import DistributedDataParallel
from torch import distributed as dist

import matplotlib
# below is needed to avoid message ""Backend TkAgg is interactive backend. Turning interactive mode on"
matplotlib.use('TkAgg',force=True)
import matplotlib.pyplot as plt

from nanugpt import utils
from nanugpt import common
from nanugpt import glogging as logging
from nanugpt.config import Config
from nanugpt.stopwatch import StopWatch

"""
The goal of this script to figure out the maximum throughput of model.
We first find out max device_batch_size that fits into GPU memory.
Then we plot throuput vs grad acc steps.
"""

sw  = StopWatch()

def infinite_batches(train_loader)->Iterator[Tuple[torch.Tensor, torch.Tensor]]:
    """Generate infinite batches repeatedly from train_loader"""
    batch_iter = iter(train_loader)
    while True:
        try:
            x, y = next(batch_iter)
            yield x, y
        except StopIteration:
            batch_iter = iter(train_loader)


def train_step(batch_iter, model, gread_acc_steps, scaler, optimizer, grad_clip,
               device, torch_info, amp_ctx, get_loss):
    """Run a full step of train"""

    sw.start('train_step')

    step_samples, step_tokens = 0, 0

    # run forward passes
    sw.start('train_step\forward')
    for _ in range(gread_acc_steps):
        x, y = next(batch_iter)
        loss, n_samples, n_tokens = forward_xy(x, y, model, device, torch_info, amp_ctx, get_loss)
        loss = loss / gread_acc_steps # scale the loss to account for gradient accumulation
        step_samples += n_samples
        step_tokens += n_tokens
    sw.pause('train_step\forward')

    # backward pass, with gradient scaling if training in fp16
    sw.start('train_step\backward')
    scaler.scale(loss).backward() # type: ignore
    sw.pause('train_step\backward')

    # clip the gradient
    if grad_clip != 0.0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

    # step the optimizer and scaler if training in fp16
    scaler.step(optimizer)
    scaler.update()

    # flush the gradients as soon as we can, no need for this memory anymore
    optimizer.zero_grad(set_to_none=True)

    sw.pause('train_step')

    return step_samples, step_tokens

def forward_xy(x, y, model, device, torch_info, amp_ctx, get_loss):
    """Run a forward pass"""
    x, y = x.pin_memory().to(device, non_blocking=True) if torch_info.is_cuda else x.to(device), \
        y.pin_memory().to(device, non_blocking=True) if torch_info.is_cuda else y.to(device)

    n_samples = len(x)
    n_tokens = x.numel()

    with amp_ctx:
        sw.start('forward')
        logits = model(x)
        sw.pause('forward')
        loss, correct, n_preds = get_loss(logits, y)

    return loss, n_samples, n_tokens

def measure_global_batch(config:Mapping,
                     logger:logging.Logger,
                     model_warmup_steps:int=10,
                     measurement_steps:int=10,):

    optimizer_config = config['optimizer']
    train_config = config['training']
    out_dir = config['general']['out_dir']
    data_config = config['data']
    loss_config = config['loss']

    global_batch_size = train_config['global_batch_size']
    device_batch_size = train_config['device_batch_size']
    gradient_accumulation_steps = global_batch_size // device_batch_size

    # don't measure timings yet
    sw.clear_all()
    sw.enable_all(False)

    # setup torch
    device, amp_ctx, torch_info = common.setup_device(config, logger)
    assert torch_info.is_master == utils.is_master_process(), "torch_info.is_master != utils.is_master_process()"

    # setup output dir
    if torch_info.is_master:
        out_dir = utils.full_path(out_dir, create=True)
        logger.summary({'run/out_dir': out_dir})

    if torch_info.is_cuda:
        torch.cuda.empty_cache()

    # setup data
    get_data = utils.import_fn(data_config['module'])
    get_loss = utils.import_fn(loss_config['module'])
    train_loader, val_loader, test_loader = get_data(local_rank=torch_info.local_rank,
                                                    **data_config['module_kwargs'])

    # create tokenizer
    tokenizer, tokenizer_config = common.create_tokenizer(config, logger)

    # create model
    model, model_config = common.create_model(config, logger, device, vocab_size=len(tokenizer))

    # create optimizer
    get_optim = utils.import_fn(optimizer_config['module'])
    optimizer = get_optim(model,
                        enable_fused=torch_info.is_cuda,
                        **optimizer_config['module_kwargs'])

    # initialize a GradScaler. If enabled=False scaler is a no-op
    # we need loss scaling only for fp16 due to reduced precision, not bf16 or fp32
    scaler = torch.cuda.amp.GradScaler(enabled=(torch_info.pt_dtype == torch.float16))

    batch_iter = iter(infinite_batches(train_loader))

    # let's warmup
    logger.info("Warming up...")
    for _ in range(model_warmup_steps):
        train_step(batch_iter=batch_iter,
                   model=model,
                   gread_acc_steps=gradient_accumulation_steps,
                   scaler=scaler,
                   optimizer=optimizer,
                   grad_clip=train_config['grad_clip'],
                   device=device,
                   torch_info=torch_info,
                   amp_ctx=amp_ctx,
                   get_loss=get_loss)
    logger.info("Warmup done")

    sw.enable_all(True)
    sw.start('measure_global_batch')
    total_samples, total_tokens = 0, 0
    for _ in range(measurement_steps):
        step_samples, step_tokens= train_step(batch_iter=batch_iter,
                   model=model,
                   gread_acc_steps=gradient_accumulation_steps,
                   scaler=scaler,
                   optimizer=optimizer,
                   grad_clip=train_config['grad_clip'],
                   device=device,
                   torch_info=torch_info,
                   amp_ctx=amp_ctx,
                   get_loss=get_loss)
        total_samples += step_samples
        total_tokens += step_tokens

    sw.pause('measure_global_batch')
    sw.enable_all(False)

    return total_samples, total_tokens, sw.report_all(), device_batch_size, gradient_accumulation_steps

def make_plot(data:List[Tuple[int, int, float]],
              save_filepath:Optional[str]):
    # using the data in plot_data, plot throughput vs grad_acc_steps, one curve with each device_batch_size, include legends and axis labels

    # create a new figure and axes
    fig, ax = plt.subplots()

    # plot each curve
    for l, x, y, *_ in data:
        ax.plot(x, y, label=l)

    # set labels
    ax.set_xlabel('grad_acc_steps')
    ax.set_ylabel('samples_per_sec')

    # add a legend
    ax.legend()

    # save the figure
    if save_filepath:
        fig.savefig(utils.full_path(save_filepath))

def measure_throuput(config:Mapping,
                     logger:logging.Logger,
                     device_batch_range:List[int],
                     grad_acc_steps_range:List[int],
                     model_warmup_steps:int=10,
                     measurement_steps:int=10,):

    data = []
    for device_batch_size in device_batch_range:
        for gradient_accumulation_steps in grad_acc_steps_range:
            train_config = config['training']
            train_config['device_batch_size'] = device_batch_size
            train_config['global_batch_size'] = device_batch_size * gradient_accumulation_steps

            try:
                total_samples, total_tokens, timings, \
                    device_batch_size, gradient_accumulation_steps = \
                        measure_global_batch(config, logger,
                                            model_warmup_steps=model_warmup_steps,
                                            measurement_steps=measurement_steps)
                elapsed_s = timings['measure_global_batch']['elapsed_total']
            except RuntimeError as e:
                total_samples, total_tokens, timings, \
                    device_batch_size, gradient_accumulation_steps = \
                        0, 0, {}, device_batch_size, gradient_accumulation_steps
                elapsed_s = float('inf')

            throughput_samples = total_samples/elapsed_s
            throughput_tokens = total_tokens/elapsed_s
            data.append((device_batch_size, gradient_accumulation_steps, throughput_samples,
                         total_samples, total_tokens, elapsed_s, timings,
                         throughput_tokens))

            logger.info({'device_batch_size':device_batch_size,
                        'gradient_accumulation_steps':gradient_accumulation_steps,
                        'total_samples':total_samples,
                        'total_tokens':total_tokens,
                        'throughput_samples': throughput_samples,
                        'throughput_tokens': throughput_tokens,
                        'timings': timings,
                    })

    plot_save_filepath = os.path.join(utils.full_path(config['general']['out_dir'], create=True),
                                 'throughput.png')
    make_plot(data, save_filepath=plot_save_filepath)

    data_save_filepath = os.path.join(utils.full_path(config['general']['out_dir'], create=True),
                                 'throughput.yaml')
    utils.save_yaml(data, data_save_filepath)

    logger.summary({'plot_save_filepath': plot_save_filepath,
                    'data_save_filepath': data_save_filepath})


if __name__ == "__main__":
    # specify config file to use as first argument in commandline
    config = Config(default_config_filepath='configs/train_gpt2/tinystories.yaml')
    # we only care about local throughput here for now
    config['general']['enable_distributed'] = False

    # Turn off unnecessary logging
    config['training']['enable_train_log'] = False
    config['logging']['enable_wandb'] = False
    config['logging']['project_name'] = 'gpt2-throughput'
    config['logging']['run_name'] = 'throughput'

    logger = common.setup_logger(utils.is_master_process(), config)

    measure_throuput(config,
                     device_batch_range=[4, 8, 16, 32, 64, 128, 256, 512, 1024],
                     grad_acc_steps_range=[1, 2, 4, 8, 16, 32],
                     logger=logger)
