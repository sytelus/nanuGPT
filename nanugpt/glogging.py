
import sys
from typing import Any, Dict, List, Mapping, Optional, Union
from functools import partial
import logging as py_logging
import psutil
import os
import timeit

from rich.logging import RichHandler
import wandb
import torch

from nanugpt.utils import full_path, is_debugging

INFO=py_logging.INFO
WARN=py_logging.WARN
ERROR=py_logging.ERROR
DEBUF=py_logging.DEBUG

def _fmt(val:Any)->str:
    if isinstance(val, torch.Tensor):
        if val.numel() == 1:
            val = val.item()
    if isinstance(val, float):
        return f'{val:.4g}'
    return str(val)

def _dict2msg(d:Mapping[str,Any])->str:
    return ', '.join(f'{k}={_fmt(v)}' for k, v in d.items())

def create_py_logger(filepath:Optional[str]=None,
                    allow_overwrite_log:bool=False,
                    project_name:Optional[str]=None,
                    run_name:Optional[str]=None,
                    run_description:Optional[str]=None,
                    py_logger_name:Optional[str]=None,    # default is root
                    level=py_logging.INFO,
                    enable_stdout=True)->py_logging.Logger:
    py_logging.basicConfig(level=level) # this sets level for standard py_logging.info calls
    logger = py_logging.getLogger(name=py_logger_name)

    # close current handlers
    for handler in logger.handlers[:]:
        handler.close()
        logger.removeHandler(handler)

    logger.setLevel(level)

    if enable_stdout:
        ch = RichHandler(
            level = level,
            show_time = True,
            show_level = False,
            log_time_format = '%H:%M',
            show_path = False,
            keywords = highlight_metric_keywords,
        )
        logger.addHandler(ch)

    logger.propagate = False # otherwise root logger prints things again

    if filepath:
        _ = full_path(os.path.dirname(filepath), create=True) # ensure dir exists
        filepath = full_path(filepath)

        if os.path.exists(filepath) and not allow_overwrite_log:
            raise FileExistsError(f'Log file {filepath} already exists. Specify different file or passt allow_overwrite_log=True.')

        # log files gets appeneded if already exist
        # zero_file(filepath)
        # use mode='a' to append
        fh = py_logging.FileHandler(filename=full_path(filepath), mode='w', encoding='utf-8')
        fh.setLevel(level)
        fh.setFormatter(py_logging.Formatter('[%(asctime)s][%(levelname)s] %(message)s'))
        logger.addHandler(fh)

    logger.info(_dict2msg({'project_name': project_name, 'run_name': run_name}))
    logger.info(_dict2msg({'run_description': run_description}))
    logger.info(_dict2msg({'filepath': filepath}))

    return logger

std_metrics = {}
std_metrics['default'] = [
                            {"name": "elapsed_s", "summary":"max"},
                            {"name": "train/step", "step_metric":"elapsed_s", "summary":"max"},
                            {"name": "train/samples", "step_metric":"elapsed_s", "summary":"max"},
                            {"name": "train/step_samples", "step_metric":"train/step", "summary":"mean"},

                            {"name": "train/token_count", "step_metric":"train/step", "summary":"max", "goal":"max"},
                            {"name": "train/loss", "step_metric":"train/samples", "summary":"min", "goal":"min"},
                            {"name": "train/loss", "step_metric":"train/step", "summary":"min", "goal":"min"},
                            {"name": "train/best_loss", "step_metric":"train/samples", "summary":"min", "goal":"min"},
                            {"name": "train/ppl", "step_metric":"train/samples", "summary":"min", "goal":"min"},
                            {"name": "train/step_interval", "step_metric":"train/step", "summary":"mean"},

                            {"name": "train/epoch", "step_metric":"train/step", "summary":"max"},
                            {"name": "train/tokens_per_sec", "step_metric":"train/step", "summary":"mean"},
                            {"name": "train/epoch_step", "step_metric":"train/step"},

                            {"name": "train/step_loss", "step_metric":"train/step", "summary":"min", "goal":"min"},
                            {"name": "train/step_ppl", "step_metric":"train/step", "summary":"min", "goal":"min"},

                            {"name": "val/best_loss", "step_metric":"train/samples", "summary":"min", "goal":"min"},
                            {"name": "val/loss", "step_metric":"train/samples", "summary":"min", "goal":"min"},
                            {"name": "val/loss", "step_metric":"train/step", "summary":"min", "goal":"min"},
                            {"name": "val/ppl", "step_metric":"train/samples", "summary":"min", "goal":"min"},
                            {"name": "val/interval", "step_metric":"train/samples", "summary":"mean"},

                            {"name": "test/loss", "step_metric":"train/samples", "summary":"min", "goal":"min"},
                            {"name": "test/ppl", "step_metric":"train/samples", "summary":"min", "goal":"min"},

                            {"name": "lr", "step_metric":"train/step", "summary":"max"},
                            {"name": "ETA_hr", "step_metric":"train/step", "summary":"max", "goal":"min"},
                            {"name": "w_norm", "step_metric":"train/samples", "summary":"mean", "goal":"min"},
                            {"name": "transformer_tflops", "step_metric":"train/step", "summary":"mean", "goal":"max"},
                            {"name": "tokens_per_sec", "step_metric":"train/step", "summary":"mean", "goal":"max"},
                    ]
std_metrics['classification'] = std_metrics['default'] +[
                        {"name": "train/acc", "step_metric":"train/samples", "summary":"max", "goal":"max"},
                        {"name": "train/step_acc", "step_metric":"train/samples", "summary":"max", "goal":"max"},
                        {"name": "test/acc", "step_metric":"train/samples", "summary":"max", "goal":"max"},
                        {"name": "val/acc", "step_metric":"train/samples", "summary":"max", "goal":"max"},
                    ]
highlight_metric_keywords = ['train/loss=', 'val/loss=', 'train/step=']


def create_wandb_logger(project_name, run_name,
                        metrics:List[Dict[str, Any]],
                        description:Optional[str]=None):

    wandb.login() # use API key from WANDB_API_KEY env variable

    run = wandb.init(project=project_name, name=run_name,
                     save_code=True, notes=description)
    for metric in metrics:
        wandb.define_metric(**metric)
    return run

def _uninit_logger(*args, **kwargs):
    raise RuntimeError('Logger not initialized. Create Logger() first.')

summary = _uninit_logger
log_config = _uninit_logger
info = _uninit_logger
warn = _uninit_logger
error = _uninit_logger
log_sys_info = _uninit_logger
finish = _uninit_logger
all_done = _uninit_logger
flush = _uninit_logger

_logger:Optional['Logger'] = None


class Logger:
    def __init__(self, master_process:bool,
                 project_name:Optional[str]=None,
                 run_name:Optional[str]=None,
                 run_description:Optional[str]=None,
                 enable_wandb=False,
                 metrics_type='default',
                 log_dir:Optional[str]=None,
                 log_filename:Optional[str]=None,
                 allow_overwrite_log=False,
                 enable_summaries=True,
                 glabal_instance:Optional[bool]=None) -> None:

        global _logger, summary, log_config, info, warn, error, log_sys_info, finish, all_done, flush

        if glabal_instance!=False and _logger is None:
            _logger = self
            # module level methods to call global logging object
            summary = partial(Logger.summary, _logger)
            log_config = partial(Logger.log_config, _logger)
            info = partial(Logger.info, _logger)
            warn = partial(Logger.warn, _logger)
            error = partial(Logger.error, _logger)
            log_sys_info = partial(Logger.log_sys_info, _logger)
            finish = partial(Logger.finish, _logger)
            all_done = partial(Logger.all_done, _logger)
            flush = partial(Logger.flush, _logger)

        self.start_time = timeit.default_timer()
        self._py_logger = None
        self._wandb_logger = None
        self.enable_wandb = enable_wandb
        self.master_process = master_process
        self.enable_summaries = enable_summaries

        if master_process:
            if log_dir or log_filename:
                log_filepath = os.path.join(full_path(str(log_dir), create=True), str(log_filename))
            else:
                log_filepath = None
            self._py_logger = create_py_logger(filepath=log_filepath,
                                            allow_overwrite_log=allow_overwrite_log,
                                            project_name=project_name,
                                            run_name=run_name,
                                            run_description=run_description)

        if enable_wandb and master_process:
            if is_debugging():
                self._py_logger.warn('Wandb logging is disabled in debug mode.')
            else:
                self._wandb_logger = create_wandb_logger(project_name, run_name,
                                                std_metrics[metrics_type],
                                                run_description)
        # else leave things to None

    def log_config(self, config):
        if not self.enable_summaries:
            return
        if self._py_logger is not None:
            self._py_logger.info(_dict2msg({'project_config': config}))
        if self.enable_wandb and self._wandb_logger is not None:
            self._wandb_logger.config.update(config)

    def info(self, d:Union[str, Mapping[str,Any]], py_logger_only:bool=False):
        if self._py_logger is not None:
            if isinstance(d, Mapping):
                d = _dict2msg(d)
            self._py_logger.info(d)

        if not py_logger_only and self.enable_wandb and self._wandb_logger is not None:
            if isinstance(d, str):
                wandb.alert(title=d[:64], text=d, level=wandb.AlertLevel.INFO)
            else:
                wandb.log(d)

    def warn(self, d:Union[str, Mapping[str,Any]], py_logger_only:bool=False,
             exception_instance:Optional[Exception]=None, stack_info:bool=False):

        if isinstance(d, Mapping):
            d = _dict2msg(d)

        if self._py_logger is not None:
            self._py_logger.warn(d, exc_info=exception_instance, stack_info=stack_info)

        if not py_logger_only and self.enable_wandb and self._wandb_logger is not None:
            ex_msg = d
            if exception_instance is not None:
                ex_msg = f'{d}\n{exception_instance}'
            wandb.alert(title=d[:64], text=ex_msg, level=wandb.AlertLevel.WARN)
        # else do nothing

    def error(self, d:Union[str, Mapping[str,Any]], py_logger_only:bool=False,
              exception_instance:Optional[Exception]=None, stack_info:bool=True):

        if isinstance(d, Mapping):
            d = _dict2msg(d)

        if self._py_logger is not None:
            self._py_logger.error(d, exc_info=exception_instance, stack_info=stack_info)

        if not py_logger_only and self.enable_wandb and self._wandb_logger is not None:
            ex_msg = d
            if exception_instance is not None:
                ex_msg = f'{d}\n{exception_instance}'
            wandb.alert(title=d[:64], text=ex_msg, level=wandb.AlertLevel.ERROR)

    def summary(self, d:Mapping[str,Any], py_logger_only:bool=False):
        if not self.enable_summaries:
            return
        if self._py_logger is not None:
            self.info(d, py_logger_only=True)

        if not py_logger_only and self.enable_wandb and self._wandb_logger is not None:
            for k, v in d.items():
                self._wandb_logger.summary[k] = v
        # else do nothing

    def log_artifact(self, name:str, type:str, file_or_dir:Optional[str], desc_markdown:Optional[str]=None, py_logger_only:bool=False):
        if self._py_logger is not None:
            self._py_logger.info(f'Artifact {type} {name}: path={file_or_dir}, desc={desc_markdown}')

        if not py_logger_only and self.enable_wandb and self._wandb_logger is not None:
            artifact = wandb.Artifact(name=name, type=type, description=desc_markdown)
            if file_or_dir:
                if os.path.isdir(file_or_dir):
                    artifact.add_dir(file_or_dir)
                else:
                    artifact.add_file(file_or_dir)
            self._wandb_logger.log_artifact(artifact)

    def log_sys_info(self):
        self.summary({
                        'device_name': torch.cuda.get_device_name() if torch.cuda.is_available() else '<not_cuda>',
                        'torch_version': torch.__version__,
                        'python_version': f'{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}',
                        'env_rank': os.environ.get('RANK', None),
                        'env_local_rank': os.environ.get('LOCAL_RANK', None),
                        'env_world_size': os.environ.get('WORLD_SIZE', None),
                        'env_master_addr': os.environ.get('MASTER_ADDR', None),
                        'env_master_port': os.environ.get('MASTER_PORT', None),
                        'env_OMP_NUM_THREADS': os.environ.get('OMP_NUM_THREADS', None),
                        'env_CUDA_HOME': os.environ.get('CUDA_HOME', None),

                        'torch.distributed.is_initialized': torch.distributed.is_initialized(),
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
        if self._wandb_logger is not None:
            self._wandb_logger.finish()
        if self._py_logger is not None:
            py_logging.shutdown()

    def all_done(self, exit_code:int=0, write_total_time:bool=True):
        if write_total_time:
            self.summary({'start_time': self.start_time, 'total_time': timeit.default_timer() - self.start_time})

        self.finish()
        exit(exit_code)

    def flush(self):
        if self._py_logger is not None:
            for handler in self._py_logger.handlers:
                handler.flush()

