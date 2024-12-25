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
from nanugpt import lin_predictor
from nanugpt.scalers.scaler_base import ScalerBase

def estimate_loss(model:torch.nn.Module, get_loss:Callable,
                  data_loader, eval_iters:Optional[int],
                  amp_ctx, is_cuda:bool, device)->Tuple[float, float]:
    model.eval()
    with torch.no_grad():
        loss_sum, correct_sum, preds_count, sample_count = 0., 0, 0, 0
        for i, (x, y) in enumerate(data_loader):
            if eval_iters is not None and i >= eval_iters: # eval_iters is None means eval the whole dataset
                break
            x, y = x.to(device, non_blocking=True) if is_cuda else x.to(device), \
                y.to(device, non_blocking=True) if is_cuda else y.to(device)
            with amp_ctx:
                loss, correct, n_preds = get_loss(model(x), y)
                n_samples = len(y)
                loss_sum += loss.item() * n_samples # loss is average so we need to multiply by n_samples to get total loss over batch
                correct_sum += correct.item()
                preds_count += n_preds
                sample_count += n_samples
    model.train()
    assert sample_count > 0 and preds_count > 0, "No samples in the dataset"
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
    start_time = timeit.default_timer()
    global_batch_size = config['training']['global_batch_size']
    project_name = config['logging']['project_name']
    run_name = config['logging']['run_name']
    device_batch_size = config['training']['device_batch_size']
    max_steps = config['training']['max_steps']
    grad_clip = config['training']['grad_clip']
    enable_train_log = config['training']['enable_train_log']
    train_log_every = config['training']['log_every']
    eval_every = config['eval']['eval_every']
    eval_iters = config['eval']['eval_iters']
    save_checkpoint = config['eval']['save_checkpoint']
    checkpoint_every_hr = config['eval']['checkpoint_every_hr']
    checkpoint_keep_best = config['eval']['checkpoint_keep_best']

    checkoint_after = config['eval']['checkoint_after']
    out_dir = config['general']['out_dir']
    data_config = config['data']
    context_length = config['model']['module_kwargs']['context_length'] #TODO: refactor so trainer is independent of context length?
    optimizer_config = config['optimizer']
    scheduler_config = config['scheduler']
    loss_config = config['loss']
    scaler_config = config['scaler']

    get_data = utils.import_fn(data_config['module'])
    get_optim = utils.import_fn(optimizer_config['module'])
    get_scheduler = utils.import_fn(scheduler_config['module'])
    get_loss = utils.import_fn(loss_config['module'])
    get_scaler = utils.import_fn(scaler_config['module'])

    # setup system, device, logger, torch
    own_logger = logger is None
    logger = common.setup_logger(config=config, logger=logger)

    device, amp_ctx, torch_info = common.setup_device(config, logger)
    assert torch_info.is_master == utils.is_master_process(), "torch_info.is_master != utils.is_master_process()"

    # compute needed accumulation steps
    grad_acc_steps = utils.calc_grad_acc(global_batch_size, device_batch_size, torch_info.world_size)
    # readjust global batch size
    global_batch_size = grad_acc_steps * device_batch_size * torch_info.world_size
    local_batch_size = grad_acc_steps * device_batch_size

    logger.summary({
                    "run/grad_acc_steps": grad_acc_steps,
                    "run/device_batch_size": device_batch_size,
                    "run/global_batch_size": global_batch_size,
                    "run/local_batch_size": local_batch_size,
                    "run/tokens_per_step": global_batch_size * context_length,
                    "run/max_steps": max_steps,
                    "run/start_time": start_time,
                    })

    # get dataset
    train_loader, val_loader, test_loader = get_data(**data_config['module_kwargs'])
    train_batch_count = len(train_loader)
    logger.summary({
                    'data/train_dataset_tokens': train_loader.dataset.token_count(),
                    'data/val_dataset_tokens': val_loader.dataset.token_count(),
                    'data/test_dataset_tokens': test_loader.dataset.token_count() if test_loader is not None else 0,
                    'data/train_dataset_len': len(train_loader.dataset),
                    'data/val_dataset_len': len(val_loader.dataset),
                    'data/test_dataset_len': len(test_loader.dataset) if test_loader is not None else 0,
                    'data/train_dataloader_len': len(train_loader),
                    'data/val_dataloader_len': len(val_loader),
                    'data/test_dataloader_len': len(test_loader) if test_loader is not None else 0
                    })

    # create tokenizer
    tokenizer, tokenizer_config = common.create_tokenizer(config, logger)
    logger.summary({'run/vocab_size': len(tokenizer)})

    # create model
    model, model_config = common.create_model(config, logger, device, vocab_size=len(tokenizer))
    model_kwargs = model_config['module_kwargs']
    n_all, n_trainable, n_embedding, n_non_embedding_trainable = utils.module_params_count(model)
    context_length = model_kwargs['context_length']
    device_step_flops = utils.transformer_flops(batch_size=local_batch_size,
        params_nonembedding_trainable=n_non_embedding_trainable,
        context_length=context_length,
        n_embd=model_kwargs['n_embd'], n_layer=model_kwargs['n_layer']
    )
    logger.summary({'model/params_all': n_all,
                    'model/params_non_emb': n_all-n_embedding,
                    'model/params_emb': n_embedding,
                    'model/params_train': n_trainable,
                    'model/params_non_emb_train': n_non_embedding_trainable,
                    'model/context_length': context_length,
                    'model/device_step_flops': device_step_flops,
                   })

    # optimizer
    optimizer = get_optim(model,
                          enable_fused=torch_info.is_cuda,
                          **optimizer_config['module_kwargs'])

    # note that model should be initialized before call to DDP
    # as DDP broadcasts initial weight from rank 0 to all other ranks
    if torch_info.is_distributed:
        model = DistributedDataParallel(model,
                                        device_ids=[torch_info.device_id],
                                        gradient_as_bucket_view=True,) # grads are kept in reducer buckets avoiding 2x memory usage

    # scheduler provides warmup and then constant lr
    scheduler = get_scheduler(optimizer, **scheduler_config['module_kwargs'])

    # initialize a GradScaler. If enabled=False scaler is a no-op
    # we need loss scaling only for fp16 due to reduced precision, not bf16 or fp32
    scaler:ScalerBase = get_scaler(torch_info)

    if torch_info.is_master:
        out_dir = utils.full_path(out_dir, create=True)
        logger.summary({'run/out_dir': out_dir})

    step, eval_count, total_samples, total_tokens = 0, 0, 0, 0
    best_train_loss, best_val_loss = float('inf'), float('inf')
    best_train_loss_step, best_val_loss_step = -1, -1
    last_checkpoint_time = timeit.default_timer()
    prev_train_losses, loss_inversions, loss_improvement_steps = [], 0,0
    max_previous_losses, pred_loss = 300, float('inf')
    checkpoint_log = []
    loop_start_time = last_eval_time = timeit.default_timer()
    batches = Batches(train_loader)
    train_time_hr = 0.0

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
                model.require_backward_grad_sync = (micro_step == grad_acc_steps - 1) # type: ignore

            # note that we don't take context length into account here
            # if we have N tokens in dataset, we move window N times and do
            # N iterations per epoch. So, token count in traditional sense is
            # number of samples, not number of tokens.
            n_samples = len(x)
            step_sample_count += n_samples
            step_token_count += x.numel()

            with amp_ctx:
                # logits = model(x)
                loss, correct, n_preds = get_loss(model(x), y)

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
            scaler.backward(loss) # type: ignore
        # --- end of gradient accumulation loop ---

        # clip the gradients
        pre_clip_norm = scaler.clip(model, optimizer, grad_clip)
        # step the optimizer (if grad were unscaled then scaler remembers and doesn't unscale again)
        scaler.step(optimizer)
        # update the scale for next iteration
        scaler.update()
        scheduler.step()
        # flush the gradients as soon as we can, no need for this memory anymore
        optimizer.zero_grad(set_to_none=True)

        fwd_bwd_interval = timeit.default_timer() - step_start_time

        # gather matrics from all ranks
        if torch_info.is_distributed:
            # dist.barrier() # not needed as reduce will sync all processes
            # reduce tensors to global_rank 0 to get numbers from all ranks
            fp32_dist = torch.tensor([loss_sum, fwd_bwd_interval, pre_clip_norm,
                                      correct_sum, step_preds_count, step_sample_count, step_token_count], dtype=torch.float32, device=device)
            dist.reduce(fp32_dist, dst=0, op=dist.ReduceOp.SUM)
            if torch_info.is_master:
                loss_sum,fwd_bwd_interval_sum, pre_clip_norm_sum, \
                correct_sum, step_preds_count, step_sample_count, step_token_count = tuple(fp32_dist.tolist())
                # use sum of all worker values so we have more accurate idea of outliers
                fwd_bwd_interval, pre_clip_norm = fwd_bwd_interval_sum, pre_clip_norm_sum
                # convert back to int
                correct_sum, step_preds_count, step_sample_count, step_token_count = int(correct_sum), int(step_preds_count), int(step_sample_count), int(step_token_count)

        total_samples += step_sample_count
        total_tokens += step_token_count
        train_acc = correct_sum / step_preds_count
        train_loss = loss_sum / step_sample_count
        if train_loss < prev_train_losses[-1] if prev_train_losses else float('inf'):
            loss_inversions += 1
        prev_train_losses[-max_previous_losses:] += [train_loss]
        if train_loss < best_train_loss:
            best_train_loss = train_loss
            best_train_loss_step = step
            loss_improvement_steps += 1

        # update iteration metrics
        if torch_info.is_master:
            elapsed_hr = (timeit.default_timer() - loop_start_time)/3600.0
            loss_pred_model = lin_predictor.fit(list(range(step-len(prev_train_losses)+1, step+1)), prev_train_losses)
            pred_loss = float(lin_predictor.predict(loss_pred_model, [max_steps-1])[0])
            step_interval = timeit.default_timer() - step_start_time
            train_time_hr += step_interval / 3600.0
            run_flops = utils.transformer_flops(batch_size=total_samples,
                params_nonembedding_trainable=n_non_embedding_trainable,
                context_length=context_length,
                n_embd=model_kwargs['n_embd'], n_layer=model_kwargs['n_layer']
            )
            metrics.update({
                "train/step": step,
                "train/loss": train_loss,
                "train/acc": train_acc,
                "train/ppl": math.exp(train_loss),
                "train/best_loss": best_train_loss,
                "train/best_loss_step": best_train_loss_step,
                "train/epoch": step * grad_acc_steps / train_batch_count,
                "train/step_interval": step_interval,
                "train/train_time_hr": train_time_hr,
                "train/fwd_bwd_interval": fwd_bwd_interval,
                "train/samples": total_samples,
                "train/step_samples": step_sample_count,
                "train/tokens": total_tokens,
                "train/tokens_per_sec": step_token_count / fwd_bwd_interval,
                "train/loss_inversions": 100.0*loss_inversions/(step+1),
                "train/loss_improvement_steps": 100.0*loss_improvement_steps/(step+1),
                "train/pred_loss": pred_loss,
                "train/pre_clip_norm": pre_clip_norm,
                "run/lr": optimizer.param_groups[0]['lr'],
                "run/flops": run_flops,
                "run/elapsed_hr": elapsed_hr,
                "run/eta_hr": elapsed_hr * (max_steps-step-1) / (step+1),
                "run/checkpoint_since_hr": (timeit.default_timer() - last_checkpoint_time)/3600.0,
            })

        # is it time to evaluate? We evaluate after 1st step to get initial loss.
        eval_performed = False
        if torch_info.is_master and ((step+1) % eval_every == 0 or step+1 >= max_steps):
            max_memory_allocated = torch.cuda.max_memory_allocated() if torch_info.is_cuda else 0

            torch.cuda.empty_cache() # clear cache before evaluation

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
                "val/generalization_gap": val_loss - train_loss,
                "val/acc": val_acc,
                "val/ppl": math.exp(val_loss),
                "val/best_loss": best_val_loss,
                "val/best_loss_step": best_val_loss_step,
                "run/w_norm": utils.weight_norm(model),
                "val/interval": eval_interval,
                "train/max_memory_allocated": max_memory_allocated,
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

            torch.cuda.empty_cache() # clear cache after evaluation

        # if this is last step or enough time has passed, save checkpoint
        can_checkpoint = save_checkpoint and \
                ((step+1 >= max_steps) or \
                    (step > checkoint_after and \
                        (timeit.default_timer() - last_checkpoint_time) / 3600.0 > checkpoint_every_hr)
                )

        # consolidate optim state
        if can_checkpoint:
            # distributed optimizer like ZeroOptimizer needs to be consolidated before saving
            # if optimizer has method called `consolidate_state_dict` then call it
            # this needs to be run on all ranks
            if hasattr(optimizer, 'consolidate_state_dict'):
                optimizer.consolidate_state_dict()

        # save checkpoint only on master
        if torch_info.is_master and can_checkpoint:
            # TODO: below is only for debugging, remove later
            logger.info({"step": step, "max_steps": max_steps,
                         "checkpoint_since_hr": (timeit.default_timer() - last_checkpoint_time)/3600.0,
                        "checkpoint_every_hr": checkpoint_every_hr})
            last_checkpoint_time = timeit.default_timer()
            checkpoint_filename = "checkpoint_" + \
                f"{step}" if not checkpoint_keep_best else "best"
            checkpoint_filepath = utils.save_checkpoint(out_dir, checkpoint_filename,
                                    model.module if torch_info.is_distributed else model,
                                    optimizer, scheduler, step, best_val_loss)

            metrics.update({"checkpoint_filepath": checkpoint_filepath})

            checkpoint_log.append(metrics)
            logger.log_artifact(name=checkpoint_filename, type='file', file_or_dir=checkpoint_filepath)


        # Decide if we should log
        can_log = len(metrics) > 0 and torch_info.is_master and (
                        (enable_train_log and (
                            step % train_log_every == 0 or step+1 >= max_steps)
                        ) or eval_performed)
        if can_log:
            logger.info(metrics)

        # Ensure all CUDA operations are complete
        # not needed as reduce will cause sync
        # torch.cuda.synchronize()

        # master might take longer, so we need to sync before barrier
        dist.barrier() # wait for all processes to come togather

        step += 1
        if step >= max_steps:
            break


    if torch_info.is_master:
        checkpoint_log_filepath = os.path.join(out_dir, "checkpoint_log.yaml")
        utils.save_yaml(checkpoint_log, checkpoint_log_filepath)
        logger.log_artifact('checkpoint_log', 'file', file_or_dir=checkpoint_log_filepath)
        end_time = timeit.default_timer()
        logger.summary({
            'run/end_time': end_time,
            'run/total_time_hr': (end_time - start_time)/3600.0
        })


        if torch_info.is_distributed:
            dist.destroy_process_group()

        if own_logger:
            logger.shutdown()
