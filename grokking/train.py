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
    train_loader, val_loader, test_loader, tokenizer = get_data(
        config['operation'],
        config['prime'],
        config['training_fraction'],
        config['val_fraction'],
        config['batch_size'],
        config['eval_batch_size'],
        config['data_loader_seed'],
    )

    # create model
    model = Transformer(
        num_layers=config['num_layers'],
        dim_model=config['dim_model'],
        num_heads=config['num_heads'],
        num_tokens=len(tokenizer),
        seq_len=5, # currently each input eq has [eos a op b =] which is 5 tokens
        ).to(device)

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

            if step % 20 == 0 or step+1 >= num_steps:
                metrics = {
                    "train/step": step,
                    "train/acc": acc.item(),
                    "train/loss": loss.item(),
                }
                logger.info(metrics)

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
                    "test/acc": val_acc,
                    "test/loss": val_loss,
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

