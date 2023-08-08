from math import ceil
import time
import torch

from grokking.data import get_data
from grokking.model import Transformer
from grokking.logger import Logger, DEFAULT_WANDB_METRICS
from grokking import utils
from grokking.utils import ExponentialMovingAverage, SmoothedDyDx

@torch.no_grad()
def evaluate(model, val_loader, device, criterion):
    # Set model to evaluation mode
    model.eval()

    correct = 0
    loss = 0.

    # Loop over each batch from the validation set
    for batch in val_loader:
        inputs, labels = tuple(t.to(device) for t in batch)

        output = model(inputs)[-1,:,:]
        correct += (torch.argmax(output, dim=1) == labels).sum()
        loss += criterion(output, labels) * len(labels)

    loss = loss / len(val_loader.dataset)
    acc = correct / len(val_loader.dataset)

    model.train()

    return loss, acc


def train(config:dict):
    if not config['device']:
        device_name = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device_name = config['device']

    utils.setup_torch()
    utils.setup_seed(config['seed'])

    logger = Logger(config['use_wandb'], master_process=True,
                    wandb_project='grokking', wandb_run_name=None,
                    config=config,
                    wandb_metrics=DEFAULT_WANDB_METRICS + [
                        {"name": "train/acc", "step_metric":"train/step", "summary":"max", "goal":"max"},
                        {"name": "val/acc", "step_metric":"train/step", "summary":"max", "goal":"max"},
                        {"name": "lr", "step_metric":"train/step", "summary":"max", "goal":"max"},
                        {"name": "ETA_hr", "step_metric":"train/step", "summary":"max", "goal":"max"},
                        {"name": "w_norm", "step_metric":"train/step", "summary":"min", "goal":"min"},
                        {"name": "train/d_loss", "step_metric":"train/step", "summary":"min", "goal":"min"},
                        {"name": "val/d_loss", "step_metric":"train/step", "summary":"min", "goal":"min"},
                        {"name": "train/ewa_loss", "step_metric":"train/step", "summary":"min", "goal":"min"},
                        {"name": "val/ewa_loss", "step_metric":"train/step", "summary":"min", "goal":"min"},
                    ])

    logger.summary(config)

    device = torch.device(device_name)
    num_steps = config['num_steps']
    eval_every = config['eval_every']

    # get dataset
    train_loader, val_loader = get_data(
        config['operation'],
        config['prime'],
        config['training_fraction'],
        config['batch_size'],
    )

    # create model
    model = Transformer(
        num_layers=config['num_layers'],
        dim_model=config['dim_model'],
        num_heads=config['num_heads'],
        num_tokens=config['prime'] + 2,
        seq_len=5
        ).to(device)

    logger.summary({"device_name": device_name,
                    'train_data_len': len(train_loader.dataset),
                    'val_data_len': len(val_loader.dataset),
                    'train_loader_len': len(train_loader),
                    'val_loader_len': len(val_loader),
                    'epochs': ceil(num_steps / len(train_loader)),
                    'model_params': model.get_num_params(True),
                    'model_params_all': model.get_num_params(False),})

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
    ewa_train_loss, ewa_val_loss = ExponentialMovingAverage(), ExponentialMovingAverage()
    d_train_loss, d_val_loss = SmoothedDyDx(), SmoothedDyDx()
    criterion = torch.nn.CrossEntropyLoss()
    model.train()
    while step < num_steps:
        # Loop over each batch from the training set
        for batch in train_loader:
            inputs, labels = tuple(t.to(device) for t in batch)

            optimizer.zero_grad()

            # model output is tensor [4,batch_size,prime+2]
            output = model(inputs)[-1,:,:]
            loss = criterion(output, labels)
            acc = (torch.argmax(output, dim=1) == labels).sum() / len(labels)

            # Backward pass
            loss.backward()

            # Update weights
            optimizer.step()
            scheduler.step()

            ewa_train_loss.add(loss.item())
            d_train_loss.add(loss.item(), step)

            if step % 20 == 0 or step+1 >= num_steps:
                metrics = {
                    "train/step": step,
                    "train/acc": acc.item(),
                    "train/loss": loss.item(),
                    "train/ewa_loss": ewa_train_loss.value,
                    "train/d_loss": d_train_loss.value,
                }
                logger.info(metrics)

            if step % eval_every == 0 or step+1 >= num_steps:
                val_loss, val_acc = evaluate(model, val_loader, device, criterion)

                ewa_val_loss.add(loss.item())
                d_val_loss.add(loss.item(), step)

                val_metrics = {
                    "train/step": step,
                    "val/acc": val_acc.item(),
                    "val/loss": val_loss.item(),
                    "w_norm": model.weight_norm(),
                    "ETA_hr": (time.time() - start_time) / (step+1) * (num_steps - step) / 3600,
                    "lr": optimizer.param_groups[0]['lr'],
                    "val/ewa_loss": ewa_val_loss.value,
                    "val/d_loss": d_val_loss.value,
                }
                logger.info(val_metrics)

            step += 1
            if step >= num_steps:
                break

    logger.summary({"train_time": (time.time() - start_time)/3600,
                    "step_time": (time.time() - start_time)/step,})
    logger.finish()