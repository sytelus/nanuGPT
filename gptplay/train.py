from math import ceil
import time
from typing import Mapping, Tuple
import os

import numpy as np

import torch

from gptplay.datasets.grokking_data import get_data
from gptplay.optimizers.adam_w import get_optim
from gptplay.schedulers.linear import get_scheduler
from gptplay.models.tiny_transformer import TinyTransformer
from gptplay import utils

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
    device_name = config['general']['device']
    seed = config['general']['seed']
    num_steps = config['training']['num_steps']
    eval_every = config['eval']['eval_every']
    out_dir = config['general']['out_dir']
    data_config = config['data']
    model_config = config['model']
    optimizer_config = config['optimizer']
    scheduler_config = config['scheduler']
    train_log_every = config['training']['log_every']


    if not device_name:
        device_name = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device_name)

    utils.setup_sys(seed)
    utils.setup_torch(seed=seed, enable_cuda='cuda' in device_name)

    out_dir = utils.full_path(out_dir, create=True)

    # get dataset
    train_loader, val_loader, test_loader, tokenizer = get_data(**data_config)

    # create model
    model = TinyTransformer(
        num_tokens=len(tokenizer),
        **model_config).to(device)

    # optimizer
    optimizer = get_optim(model.parameters(), **optimizer_config)

    # scheduler provides warmup and then constant lr
    scheduler = get_scheduler(optimizer, **scheduler_config)

    step = 0
    epoch, epoch_step = 0, 0 # epoch steps is useful to know how many epochs we did
    criterion = torch.nn.CrossEntropyLoss()
    model.train()

    # run steps
    while step < num_steps:
        epoch_step = 0
        # Loop over the training set
        for batch in train_loader:
            inputs, labels = tuple(t.to(device) for t in batch)

            optimizer.zero_grad()

            # model output is tensor [seq_len, batch_size, token_count]
            # we only take the last token of the output for loss
            output = model(inputs)[-1,:,:]
            loss = criterion(output, labels)
            acc = (torch.argmax(output, dim=1) == labels).sum() / len(labels)

            # Backward pass
            loss.backward()

            # Update weights
            optimizer.step()
            scheduler.step()

            if step % train_log_every == 0 or step+1 >= num_steps:
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
                    "train/step": step,
                    "val/acc": val_acc,
                    "val/loss": val_loss,
                    "test/acc": test_acc,
                    "test/loss": test_loss,
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

