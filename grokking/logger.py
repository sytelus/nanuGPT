
from typing import Any, Dict, List, Mapping, Optional, Union
import logging
import psutil
import os

import wandb
import torch

from grokking.utils import full_path, is_debugging


def create_py_logger(filepath:Optional[str]=None,
                  name:Optional[str]=None,
                  level=logging.INFO,
                  enable_stdout=True)->logging.Logger:
    logging.basicConfig(level=level) # this sets level for standard logging.info calls
    logger = logging.getLogger(name=name)

    # close current handlers
    for handler in logger.handlers[:]:
        handler.close()
        logger.removeHandler(handler)

    logger.setLevel(level)

    if enable_stdout:
        ch = logging.StreamHandler()
        ch.setLevel(level)
        ch.setFormatter(logging.Formatter('%(asctime)s %(message)s', '%H:%M'))
        logger.addHandler(ch)

    logger.propagate = False # otherwise root logger prints things again

    if filepath:
        filepath = full_path(filepath)
        # log files gets appeneded if already exist
        # zero_file(filepath)
        fh = logging.FileHandler(filename=full_path(filepath))
        fh.setLevel(level)
        fh.setFormatter(logging.Formatter('[%(asctime)s][%(levelname)s] %(message)s'))
        logger.addHandler(fh)
    return logger


DEFAULT_WANDB_METRICS = [
                            {"name": "train/step"},
                            {"name": "train/loss", "step_metric":"train/step", "summary":"min", "goal":"min"},
                            {"name": "val/loss", "step_metric":"train/step", "summary":"min", "goal":"min"},
                        ]

def create_wandb_logger(wandb_project, wandb_run_name, config,
                        metrics:List[Dict[str, Any]]):

    wandb.login() # use API key from WANDB_API_KEY env variable

    run = wandb.init(project=wandb_project, name=wandb_run_name, config=config,
                     save_code=True)
    for metric in metrics:
        wandb.define_metric(**metric)
    return run

def _fmt(val:Any)->str:
    if isinstance(val, torch.Tensor):
        if val.numel() == 1:
            val = val.item()
    if isinstance(val, float):
        return f'{val:.4g}'
    return str(val)

class Logger:
    def __init__(self, enable_wandb:bool, master_process:bool,
                 wandb_project:str, wandb_run_name:Optional[str], config:dict,
                 wandb_metrics=DEFAULT_WANDB_METRICS) -> None:
        self._logger = None
        self._run = None
        self.enable_wandb = enable_wandb
        self.master_process = master_process

        if master_process:
            self._logger = create_py_logger()
        if enable_wandb and master_process and not is_debugging():
            self._run = create_wandb_logger(wandb_project, wandb_run_name, config, wandb_metrics)
        # else leave things to None

    def info(self, d:Union[str, Mapping[str,Any]], py_logger_only:bool=False):
        if self._logger is not None:
            if isinstance(d, str):
                self._logger.info(d)
            else:
                msg = ', '.join(f'{k}={_fmt(v)}' for k, v in d.items())
                self._logger.info(msg)

        if not py_logger_only and self.enable_wandb and self._run is not None:
            if isinstance(d, str):
                wandb.log({'msg': d})
            else:
                wandb.log(d)
        # else do nothing

    def summary(self, d:Mapping[str,Any], py_logger_only:bool=False):
        if self._logger is not None:
            self.info(d, py_logger_only=True)

        if not py_logger_only and self.enable_wandb and self._run is not None:
            for k, v in d.items():
                self._run.summary[k] = v
        # else do nothing

    def log_sys_info(self):
        self.summary({  'torch.distributed.is_initialized': torch.distributed.is_initialized(),
                        'torch.distributed.is_available': torch.distributed.is_available(),
                        'gloo_available': torch.distributed.is_gloo_available(),
                        'mpi_available': torch.distributed.is_mpi_available(),
                        'nccl_available': torch.distributed.is_nccl_available(),
                        'get_world_size': torch.distributed.get_world_size() if torch.distributed.is_initialized() else None,
                        'get_rank': torch.distributed.get_rank() if torch.distributed.is_initialized() else None,
                        'is_anomaly_enabled': torch.is_anomaly_enabled(),
                        'device_count': torch.cuda.device_count(),

                        'cudnn.enabled': torch.backends.cudnn.enabled,
                        'cudnn.benchmark': torch.backends.cudnn.benchmark,
                        'cudnn.deterministic': torch.backends.cudnn.deterministic,
                        'cudnn.version': torch.backends.cudnn.version(),

                        'CUDA_VISIBLE_DEVICES': os.environ['CUDA_VISIBLE_DEVICES']
                                if 'CUDA_VISIBLE_DEVICES' in os.environ else 'NotSet',

                        'memory_gb': psutil.virtual_memory().available / (1024.0 ** 3),
                        'cpu_count': psutil.cpu_count(),
                        })

    def finish(self):
        if self._run is not None:
            self._run.finish()
