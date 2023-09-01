
from typing import Any, Dict, List, Mapping, Optional, Union
import logging
import psutil
import os

import wandb
import torch

from gptplay.utils import full_path, is_debugging


def _fmt(val:Any)->str:
    if isinstance(val, torch.Tensor):
        if val.numel() == 1:
            val = val.item()
    if isinstance(val, float):
        return f'{val:.4g}'
    return str(val)

def _dict2msg(d:Mapping[str,Any])->str:
    return ', '.join(f'{k}={_fmt(v)}' for k, v in d.items())

def create_py_logger(filepath:Optional[str]=None, allow_overwrite_log:bool=False,
                  project:Optional[str]=None, run_name:Optional[str]=None,
                  run_description:Optional[str]=None,
                  project_config:Optional[Mapping]=None,
                  py_logger_name:Optional[str]=None,    # default is root
                  level=logging.INFO,
                  enable_stdout=True)->logging.Logger:
    logging.basicConfig(level=level) # this sets level for standard logging.info calls
    logger = logging.getLogger(name=py_logger_name)

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

        if os.path.exists(filepath) and not allow_overwrite_log:
            raise FileExistsError(f'Log file {filepath} already exists. Specify different file or passt allow_overwrite_log=True.')

        # log files gets appeneded if already exist
        # zero_file(filepath)
        # use mode='a' to append
        fh = logging.FileHandler(filename=full_path(filepath), mode='w', encoding='utf-8')
        fh.setLevel(level)
        fh.setFormatter(logging.Formatter('[%(asctime)s][%(levelname)s] %(message)s'))
        logger.addHandler(fh)

    logger.info(_dict2msg({'project': project, 'run_name': run_name}))
    logger.info(_dict2msg({'run_description': run_description}))
    logger.info(_dict2msg({'filepath': filepath}))
    if project_config:
        logger.info(_dict2msg({'project_config': project_config}))

    return logger


DEFAULT_WANDB_METRICS = [
                            {"name": "train/step"},
                            {"name": "train/loss", "step_metric":"train/step", "summary":"min", "goal":"min"},
                            {"name": "val/loss", "step_metric":"train/step", "summary":"min", "goal":"min"},
                        ]

def create_wandb_logger(wandb_project, wandb_run_name, project_config,
                        metrics:List[Dict[str, Any]], description:Optional[str]=None):

    wandb.login() # use API key from WANDB_API_KEY env variable

    run = wandb.init(project=wandb_project, name=wandb_run_name, config=project_config,
                     save_code=True, notes=description)
    for metric in metrics:
        wandb.define_metric(**metric)
    return run

class Logger:
    def __init__(self, master_process:bool,
                 project:Optional[str]=None, run_name:Optional[str]=None,
                 run_description:Optional[str]=None,
                 project_config:Optional[Mapping]=None,
                 enable_wandb=False,
                 wandb_metrics=DEFAULT_WANDB_METRICS,
                 log_filepath:Optional[str]=None, allow_overwrite_log=False) -> None:
        self._logger = None
        self._run = None
        self.enable_wandb = enable_wandb
        self.master_process = master_process

        if master_process:
            self._logger = create_py_logger(filepath=log_filepath, allow_overwrite_log=allow_overwrite_log,
                                            project=project, run_name=run_name, run_description=run_description,
                                            project_config=project_config)
        if enable_wandb and master_process and not is_debugging():
            self._run = create_wandb_logger(project, run_name, project_config, wandb_metrics, run_description)
        # else leave things to None

    def info(self, d:Union[str, Mapping[str,Any]], py_logger_only:bool=False):
        if self._logger is not None:
            if isinstance(d, str):
                self._logger.info(d)
            else:
                self._logger.info(_dict2msg(d))

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
        if self._logger is not None:
            logging.shutdown()

    def flush(self):
        if self._logger is not None:
            for handler in self._logger.handlers:
                handler.flush()
