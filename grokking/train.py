from math import ceil
import time
from typing import Mapping, Tuple
import os

import numpy as np

import torch

from grokking.data import get_data
from grokking.model import Transformer
from grokking.logger import Logger, DEFAULT_WANDB_METRICS
from grokking import utils
from grokking.utils import ExponentialMovingAverage, SmoothedDyDx


def evaluate(model, val_loader, device, criterion)->Tuple[float, float]:
    correct = 0
    loss_sum = 0.
    loss, acc = 0., 0.

    # Set model to evaluation mode
    model.eval()

    with torch.no_grad():
        # Loop over each batch from the validation set
        for batch in val_loader:
            inputs, labels = tuple(t.to(device) for t in batch)

            output = model(inputs)[-1,:,:]
            correct += (torch.argmax(output, dim=1) == labels).sum().item()
            loss_sum += criterion(output, labels).item() * len(labels)

    loss = loss_sum / len(val_loader.dataset)
    acc = correct / len(val_loader.dataset)

    model.train()
    return loss, acc


def train(config:Mapping, logger):
    if not config['device']:
        device_name = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device_name = config['device']

    utils.setup_torch()
    utils.setup_seed(config['seed'])

    device = torch.device(device_name)
    num_steps = config['num_steps']
    eval_every = config['eval_every']
    out_dir = utils.full_path(config['out_dir'], create=True)

    # get dataset
    start_time = time.time()
    train_loader, val_loader, test_loader, tokenizer = get_data(
        config['operation'],
        config['prime'],
        config['training_fraction'],
        config['val_fraction'],
        config['batch_size'],
        config['eval_batch_size'],
        config['data_loader_seed'],
    )
    data_gen_time = time.time() - start_time

    # create model
    model = Transformer(
        num_layers=config['num_layers'],
        dim_model=config['dim_model'],
        num_heads=config['num_heads'],
        num_tokens=len(tokenizer),
        seq_len=5, # currently each input eq has [eos a op b =] which is 5 tokens
        ).to(device)

    #torch.save(model.state_dict(), os.path.join(out_dir, 'model42_8.pt'))

    model42_8 = Transformer(
        num_layers=config['num_layers'],
        dim_model=config['dim_model'],
        num_heads=config['num_heads'],
        num_tokens=len(tokenizer),
        seq_len=5, # currently each input eq has [eos a op b =] which is 5 tokens
        ).to(device)
    model42_8.load_state_dict(torch.load(os.path.join(out_dir, 'model42_8.pt')))

    #model.load_state_dict(torch.load(os.path.join(out_dir, 'model42_8.pt')))

    modules = list(model.modules())
    modules42_8 = list(model42_8.modules())

    # modules[5].load_state_dict(modules42_8[5].state_dict())
    # modules[7].load_state_dict(modules42_8[7].state_dict())
    # modules[8].load_state_dict(modules42_8[8].state_dict())
    # modules[12].load_state_dict(modules42_8[12].state_dict())
    modules[14].load_state_dict(modules42_8[14].state_dict())
    #modules[16].load_state_dict(modules42_8[16].state_dict())
    modules[17].load_state_dict(modules42_8[17].state_dict())
    # modules[21].load_state_dict(modules42_8[21].state_dict())
    modules[1].load_state_dict(modules42_8[1].state_dict())
    #modules[2].load_state_dict(modules42_8[2].state_dict())
    #modules[4].load_state_dict(modules42_8[4].state_dict())
    #modules[13].load_state_dict(modules42_8[13].state_dict())
    #modules[22].load_state_dict(modules42_8[22].state_dict())
    modules[23].load_state_dict(modules42_8[23].state_dict())



    # for i in range(len(modules)):
    #     module_str = str(modules[i]).replace('\n', '->')
    #     patam_count = sum(p.numel() for p in modules[i].parameters())
    #     param_norm = sum(p.norm().item() for p in modules[i].parameters())
    #     param_mean = sum(p.mean().item() for p in modules[i].parameters())
    #     param_norm42_8 = sum(p.norm().item() for p in modules42_8[i].parameters())
    #     param_mean42_8 = sum(p.mean().item() for p in modules42_8[i].parameters())
    #     logger.info({'module_index': i,
    #                  'patam_count': patam_count,
    #                  'param_norm': param_norm,
    #                  'param_mean': param_mean,
    #                  'param_norm42_8': param_norm42_8,
    #                  'param_mean42_8': param_mean42_8,
    #                  'module_str': module_str})
    # logger.flush()

    # weights = np.concatenate([p.detach().cpu().flatten().numpy() for p in model.get_params(non_embedding=True)])
    # np.save(os.path.join(out_dir, 'seed56_7_weights.npy'), weights)

    # optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['learning_rate'],
        betas=(0.9, 0.98),
        weight_decay=config['weight_decay']
        )

    # scheduler provides warmup and then constant lr
    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor = 1.e-8, total_iters=10
    )

    step, start_time = 0, time.time()
    epoch, epoch_step = 0, 0
    ewa_train_loss, ewa_val_loss = ExponentialMovingAverage(weight=0.1), ExponentialMovingAverage(weight=0.1)
    w_norm_ewa = ExponentialMovingAverage(weight=0.3)
    d_train_loss = SmoothedDyDx(y_ema_weight=1.0, x_ema_weight=0.5, dx_ema_weight=0.9,
                                 dy_ema_weight=1.0, dydx_ema_weight=0.1)
    d_val_loss = SmoothedDyDx(y_ema_weight=1.0, x_ema_weight=0.5, dx_ema_weight=0.9,
                                 dy_ema_weight=1.0, dydx_ema_weight=0.1)
    criterion = torch.nn.CrossEntropyLoss()
    model.train()
    while step < num_steps:
        epoch_step = 0
        # Loop over each batch from the training set
        for batch in train_loader:
            inputs, labels = tuple(t.to(device) for t in batch)

            optimizer.zero_grad()

            # model output is tensor [5,batch_size,prime+2]
            # [EOS a op b =] is input to model which is 5 tokens
            # output is [a op b = c] which is 5 tokens
            # we only take the last token of the output for loss
            output = model(inputs)[-1,:,:]
            loss = criterion(output, labels)
            acc = (torch.argmax(output, dim=1) == labels).sum() / len(labels)

            # Backward pass
            loss.backward()

            # Update weights
            optimizer.step()
            scheduler.step()

            # ewa_train_loss.add(loss.item())
            # d_train_loss.add(loss.item(), step)

            # if step % 20 == 0 or step+1 >= num_steps:
            #     metrics = {
            #         "train/step": step,
            #         "train/acc": acc.item(),
            #         "train/loss": loss.item(),
            #         "train/ewa_loss": ewa_train_loss.value,
            #         "train/d_loss": d_train_loss.value,
            #     }
                #logger.info(metrics)

            if (step+1) % eval_every == 0 or step+1 >= num_steps:
                val_loss, val_acc = evaluate(model, val_loader, device, criterion)
                if test_loader:
                    test_loss, test_acc = evaluate(model, test_loader, device, criterion)
                else:
                    test_loss, test_acc = -1, -1

                w_norm = model.weight_norm()

                val_metrics = {
                    "seed": config['seed'],
                    "data_loader_seed": config['data_loader_seed'],
                    "train/step": step,
                    "val/acc": val_acc,
                    "val/loss": val_loss,
                    "train/acc": acc.item(),
                    "train/loss": loss.item(),
                    "w_norm": w_norm,
                    "lr": optimizer.param_groups[0]['lr'],
                }
                logger.info(val_metrics)


            step += 1
            epoch_step += 1
            if step >= num_steps:
                break
        epoch += 1

