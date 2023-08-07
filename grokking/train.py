from math import ceil
import torch

from grokking.data import get_data
from grokking.model import Transformer
from grokking.logger import Logger, DEFAULT_WANDB_METRICS
from grokking import utils

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
    utils.setup_torch()
    utils.setup_seed(42)

    logger = Logger(config['use_wandb'], master_process=True,
                    wandb_project='grokking', wandb_run_name=None,
                    config=config,
                    wandb_metrics=DEFAULT_WANDB_METRICS + [
                        {"name": "train/acc", "step_metric":"train/step", "summary":"max", "goal":"max"},
                        {"name": "val/acc", "step_metric":"train/step", "summary":"max", "goal":"max"},
                    ])

    logger.summary(config)

    device = torch.device(config['device'])
    num_steps = config['num_steps']
    eval_every = config['eval_every']

    # get dataset
    train_loader, val_loader = get_data(
        config['operation'],
        config['prime'],
        config['training_fraction'],
        config['batch_size']
        )

    # create model
    model = Transformer(
        num_layers=config['num_layers'],
        dim_model=config['dim_model'],
        num_heads=config['num_heads'],
        num_tokens=config['prime'] + 2,
        seq_len=5
        ).to(device)

    num_params = sum(p.numel() for p in model.parameters())
    logger.summary({'train_data_len': len(train_loader.dataset),
                    'val_data_len': len(val_loader.dataset),
                    'train_loader_len': len(train_loader),
                    'val_loader_len': len(val_loader),
                    'model_params': num_params})

    # optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['learning_rate'],
        betas=(0.9, 0.98),
        weight_decay=config['weight_decay']
        )

    # scheduler
    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor = 0.1, total_iters=9
    )

    num_epochs = ceil(num_steps / len(train_loader))

    step = 0
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

            if step % 20 == 0 or step+1 >= num_steps:
                metrics = {
                    "step": step,
                    "train/acc": acc,
                    "train/loss": loss,
                }
                logger.info(metrics)

            if step % eval_every == 0 or step+1 >= num_steps:
                val_loss, val_acc = evaluate(model, val_loader, device, criterion)
                val_metrics = {
                    "step": step,
                    "val/acc": val_acc,
                    "val/loss": val_loss,
                }
                logger.info(val_metrics)

            step += 1
            if step >= num_steps:
                break

