from typing import Mapping, Tuple, Optional, Callable, Mapping
import os
import timeit
import math

import torch
from torch.nn.parallel import DistributedDataParallel
from torch import distributed as dist

from nanugpt import utils
from nanugpt import common
from nanugpt import glogging as logging

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

class Batches:
    def __init__(self, loader) -> None:
        self.loader = loader
        self.iter = iter(loader)

    def next(self):
        try:
            return next(self.iter)
        except StopIteration:
            self.iter = iter(self.loader)
            return next(self.iter)

def train(config:Mapping, logger:Optional[logging.Logger]=None):
    grad_acc_steps = config['training']['gradient_accumulation_steps']
    adj_grad_acc_gpu_count = config['training']['adj_grad_acc_gpu_count']
    project_name = config['logging']['project_name']
    run_name = config['logging']['run_name']
    train_batch_size = config['training']['train_batch_size']
    max_steps = config['training']['max_steps']
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
    logger = common.setup_logger(utils.is_master_process(), config, logger)

    device, amp_ctx, torch_info = common.setup_device(config, logger)
    assert torch_info.is_master == utils.is_master_process(), "torch_info.is_master != utils.is_master_process()"

    # adjust gradient accumulation steps if we are doing distributed training
    if torch_info.is_distributed and adj_grad_acc_gpu_count:
        assert grad_acc_steps % torch_info.world_size == 0, f'grad_acc_steps ({grad_acc_steps}) must be divisible by ddp_world_size ({torch_info.world_size})'
        grad_acc_steps = grad_acc_steps // torch_info.world_size

    logger.summary({
                    "run/grad_acc_steps": grad_acc_steps,
                    "run/gpu_batch_size": train_batch_size,
                    "run/global_batch_size": grad_acc_steps * train_batch_size * torch_info.world_size,
                    "run/local_batch_size": grad_acc_steps * train_batch_size,
                    "run/tokens_per_iter": grad_acc_steps * train_batch_size * torch_info.world_size * context_length
                    })

    # get dataset
    train_loader, val_loader, test_loader = get_data(local_rank=torch_info.local_rank,
                                                     **data_config['module_kwargs'])
    train_batch_count = len(train_loader)
    logger.summary({
                    'data/train_dataset_len': len(train_loader.dataset),
                    'data/val_dataset_len': len(val_loader.dataset),
                    'data/test_dataset_len': len(test_loader.dataset) if test_loader is not None else 0,
                    'data/train_dataloader_len': train_batch_count,
                    'data/val_dataloader_len': len(val_loader),
                    'data/test_dataloader_len': len(test_loader) if test_loader is not None else 0
                    })

    # create tokenizer
    tokenizer, tokenizer_config = common.create_tokenizer(config, logger)
    logger.summary({'vocab_size': len(tokenizer)})

    # create model
    model, model_config = common.create_model(config, logger, device, vocab_size=len(tokenizer))
    model_kwargs = model_config['module_kwargs']

    n_all, n_trainable, n_embedding, n_non_embedding_trainable = utils.module_params_count(model)
    logger.summary({'model/params_all': n_all,
                    'model/params_non_emb': n_all-n_embedding,
                    'model/params_emb': n_embedding,
                    'model/params_trai': n_trainable,
                    'model/params_non_emb_train': n_non_embedding_trainable,
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
        logger.summary({'run/out_dir': out_dir})

    step, eval_count, total_samples, total_tokens = 0, 0, 0, 0
    best_train_loss, best_val_loss = float('inf'), float('inf')
    best_train_loss_step, best_val_loss_step, last_checkpoint_step = -1, -1, -1
    checkpoint_log = []
    loop_start_time = last_eval_time = timeit.default_timer()
    batches = Batches(train_loader)

    # run steps
    while step < max_steps:
        step_start_time = timeit.default_timer()
        step_sample_count, step_token_count = 0, 0
        loss_sum, correct_sum, step_preds_count = 0., 0, 0
        metrics = {} # add metrics here if any

        model.train()

        x, y = batches.next()

        # grad accumulations
        for micro_step in range(grad_acc_steps):
            x, y = x.pin_memory().to(device, non_blocking=True) if torch_info.is_cuda else x.to(device), \
                y.pin_memory().to(device, non_blocking=True) if torch_info.is_cuda else y.to(device)

            if torch_info.is_distributed:
                # Instead of model.no_sync(), we do Karpathy's hack
                # On last step, flag model to require backward grad sync
                model.require_backward_grad_sync = (micro_step == grad_acc_steps - 1)

            # note that we don't take context length into account here
            # if we have N tokens in dataset, we move window N times and do
            # N iterations per epoch. So, token count in traditional sense is
            # number of samples, not number of tokens.
            n_samples = len(x)
            step_sample_count += n_samples
            step_token_count += x.numel()

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
                loss = loss / grad_acc_steps # scale the loss to account for gradient accumulation

            # below runs async while backward pass is running because dataloader is configures with workers
            x, y = batches.next()

            # backward pass, with gradient scaling if training in fp16
            scaler.scale(loss).backward() # type: ignore
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

        fwd_bwd_interval = timeit.default_timer() - step_start_time

        if torch_info.is_distributed:
            # reduce tensors to rank 0 to get numbers from all ranks
            fp32_dist = torch.tensor([loss_sum, fwd_bwd_interval], dtype=torch.float32, device=device)
            int_dist = torch.tensor([correct_sum, step_preds_count, step_sample_count, step_token_count], dtype=torch.int64, device=device)
            dist.reduce(fp32_dist, dst=0, op=dist.ReduceOp.SUM)
            dist.reduce(int_dist, dst=0,op=dist.ReduceOp.SUM)
            if torch_info.is_master:
                loss_sum,fwd_bwd_interval_sum = tuple(fp32_dist.tolist())
                fwd_bwd_interval = fwd_bwd_interval_sum
                correct_sum, step_preds_count, step_sample_count,step_token_count = tuple(int_dist.tolist())

        total_samples += step_sample_count
        total_tokens += step_token_count
        train_loss = loss_sum / step_sample_count
        train_acc = correct_sum / step_preds_count
        if train_loss < best_train_loss:
            best_train_loss = train_loss
            best_train_loss_step = step

        if torch_info.is_master:
            transformer_tflops = utils.transformer_tflops(batch_size=step_sample_count,
                params_nonembedding_trainable=n_non_embedding_trainable,
                context_length=context_length, dt=fwd_bwd_interval,
                n_embd=model_kwargs['n_embd'], n_layer=model_kwargs['n_layer'],
                forward_iters=grad_acc_steps, backward_iters=1
            )

            metrics.update({
                "train/step": step,
                "train/loss": train_loss,
                "train/acc": train_acc,
                "train/ppl": math.exp(train_loss),
                "train/best_loss": best_train_loss,
                "train/best_loss_step": best_train_loss_step,
                "train/epoch": step * grad_acc_steps / train_batch_count,
                "train/step_interval": timeit.default_timer() - step_start_time,
                "train/fwd_bwd_interval": fwd_bwd_interval,
                "train/samples": total_samples,
                "train/step_samples": step_sample_count,
                "train/tokens": total_tokens,
                "train/tokens_per_sec": step_token_count / fwd_bwd_interval,
                "lr": optimizer.param_groups[0]['lr'],
                'tflops': transformer_tflops,
                "elapsed_hr": (timeit.default_timer() - loop_start_time)/3600.0
            })

        # is it time to evaluate? We evaluate after 1st step to get initial loss.
        eval_performed = False
        if torch_info.is_master and ((step+1) % eval_every == 0 or step+1 >= max_steps):
            eval_performed = True
            eval_count += 1
            eval_interval = timeit.default_timer() - last_eval_time

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
            })
            if step+1 >= max_steps and test_loader:
                test_loss, test_acc = estimate_loss(model, get_loss, test_loader, None,
                                            amp_ctx, torch_info.is_cuda, device)
                metrics.update({
                    "test/loss": test_loss,
                    "test/ppl": math.exp(test_loss),
                    "test/acc": test_acc,
                })
            last_eval_time = timeit.default_timer()

            # if this is the best model so far, save it
            if save_checkpoint and last_checkpoint_step < best_val_loss_step and \
                    ((step+1 >= max_steps) or \
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

        # Decide if we should log
        can_log = len(metrics) > 0 and torch_info.is_master and (
                        (enable_train_log and (
                            step % train_log_every == 0 or step+1 >= max_steps)
                        ) or eval_performed)
        if can_log:
            logger.info(metrics)

        step += 1
        if step >= max_steps:
            break


    if torch_info.is_master:
        checkpoint_log_filepath = os.path.join(out_dir, "checkpoint_log.yaml")
        utils.save_yaml(checkpoint_log, checkpoint_log_filepath)
        logger.log_artifact('checkpoint_log', 'file', file_or_dir=checkpoint_log_filepath)

        if torch_info.is_distributed:
            dist.destroy_process_group()

        if own_logger:
            logger.all_done()
