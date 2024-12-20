
import sys
from typing import Any, Dict, List, Mapping, Optional, Union, Set, Iterable, Callable
from functools import partial
import logging as py_logging
import psutil
import os
import timeit
import json
import atexit

from rich.logging import RichHandler
import wandb
import torch

from nanugpt import utils

INFO=py_logging.INFO
WARN=py_logging.WARN
ERROR=py_logging.ERROR
DEBUF=py_logging.DEBUG

# when app exist, call shutdown to save everything in log system
_atexit_reg = False # is hook for atexit registered?
def install_atexit():
    # create hooks to execute code when script exits
    global _atexit_reg
    if not _atexit_reg:
        atexit.register(on_app_exit)
        _atexit_reg = True
def on_app_exit():
    print('Process exit:', os.getpid(), flush=True)
    shutdown()

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
        _ = utils.full_path(os.path.dirname(filepath), create=True) # ensure dir exists
        filepath = utils.full_path(filepath)

        if os.path.exists(filepath) and not allow_overwrite_log:
            raise FileExistsError(f'Log file {filepath} already exists. Specify different file or passt allow_overwrite_log=True.')

        # log files gets appeneded if already exist
        # zero_file(filepath)
        # use mode='a' to append
        fh = py_logging.FileHandler(filename=filepath, mode='w', encoding='utf-8')
        fh.setLevel(level)
        fh.setFormatter(py_logging.Formatter('[%(asctime)s][%(levelname)s] %(message)s'))
        logger.addHandler(fh)

    logger.info(_dict2msg({'project_name': project_name, 'run_name': run_name}))
    logger.info(_dict2msg({'run_description': run_description}))
    logger.info(_dict2msg({'filepath': filepath}))

    return logger

std_metrics = {}
std_metrics['default'] = [
                            {"name": "run/elapsed_hr", "step_metric":"train/step", "summary":"last"},
                            {"name": "run/eta_hr", "step_metric":"train/step", "summary":"last"},
                            {"name": "run/flops", "step_metric":"train/step", "summary":"last"},
                            {"name": "run/checkpoint_since_hr", "step_metric":"train/step", "summary":"last"},
                            {"name": "run/total_time_hr", "step_metric":"train/step", "summary":"last"},
                            {"name": "train/step", "step_metric":"run/elapsed_hr", "summary":"last"},
                            {"name": "train/samples_hr", "step_metric":"run/elapsed_hr", "summary":"last"},
                            {"name": "train/samples", "step_metric":"train/step", "summary":"last"},
                            {"name": "train/train_time_hr", "step_metric":"train/step", "summary":"last"},

                            {"name": "train/tokens", "step_metric":"train/step", "summary":"last", "goal":"max"},
                            {"name": "train/loss", "step_metric":"train/step", "summary":"last", "goal":"min"},
                            {"name": "train/best_loss", "step_metric":"train/step", "summary":"last", "goal":"min"},
                            {"name": "train/best_loss_step", "step_metric":"train/step", "summary":"last", "goal":"max"},
                            {"name": "train/ppl", "step_metric":"train/step", "summary":"last", "goal":"min"},
                            {"name": "train/pre_clip_norm", "step_metric":"train/step", "summary":"last"},
                            {"name": "train/step_interval", "step_metric":"train/step", "summary":"last"},
                            {"name": "train/pred_loss", "step_metric":"train/step", "summary":"last"},

                            {"name": "train/epoch", "step_metric":"train/step", "summary":"last"},
                            {"name": "train/tokens_per_sec", "step_metric":"train/step", "summary":"last"},
                            {"name": "train/fwd_bwd_interval", "step_metric":"train/step", "summary":"last"},

                            {"name": "train/step_loss", "step_metric":"train/step", "summary":"last", "goal":"min"},
                            {"name": "train/step_ppl", "step_metric":"train/step", "summary":"last", "goal":"min"},

                            {"name": "val/best_loss", "step_metric":"train/step", "summary":"last", "goal":"min"},
                            {"name": "val/val/generalization_gap", "step_metric":"train/step", "summary":"last", "goal":"min"},

                            {"name": "val/best_loss_step", "step_metric":"train/step", "summary":"last", "goal":"max"},
                            {"name": "val/loss", "step_metric":"train/step", "summary":"last", "goal":"min"},
                            {"name": "val/loss", "step_metric":"train/step", "summary":"last", "goal":"min"},
                            {"name": "val/ppl", "step_metric":"train/step", "summary":"last", "goal":"min"},
                            {"name": "val/interval", "step_metric":"train/step", "summary":"last"},

                            {"name": "test/loss", "step_metric":"train/step", "summary":"last", "goal":"min"},
                            {"name": "test/ppl", "step_metric":"train/step", "summary":"last", "goal":"min"},

                            {"name": "run/lr", "step_metric":"train/step", "summary":"last"},
                            {"name": "run/w_norm", "step_metric":"train/step", "summary":"last", "goal":"min"},
                            {"name": "run/flops", "step_metric":"train/step", "summary":"last", "goal":"max"},
                            {"name": "tokens_per_sec", "step_metric":"train/step", "summary":"last", "goal":"max"},
                    ]
std_metrics['classification'] = std_metrics['default'] +[
                        {"name": "train/acc", "step_metric":"train/step", "summary":"last", "goal":"max"},
                        {"name": "train/step_acc", "step_metric":"train/step", "summary":"last", "goal":"max"},
                        {"name": "test/acc", "step_metric":"train/step", "summary":"last", "goal":"max"},
                        {"name": "val/acc", "step_metric":"train/step", "summary":"last", "goal":"max"},
                    ]
highlight_metric_keywords = ['train/loss=', 'val/loss=', 'train/step=']


def create_wandb_logger(project_name, run_name,
                        metrics:List[Dict[str, Any]],
                        description:Optional[str]=None):

    wandb_host = os.environ.get('WANDB_HOST', None)
    wandb.login(host=wandb_host) # use API key from WANDB_API_KEY env variable

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
shutdown = _uninit_logger
flush = _uninit_logger

_logger:Optional['Logger'] = None
_except_handler_installed:bool = False

def get_logger()->'Logger':
    global _logger
    if _logger is None:
        raise RuntimeError('Logger not initialized. Create Logger() first.')
    return _logger
class Logger:
    def __init__(self, master_process:bool,
                 project_name:Optional[str]=None,
                 run_name:Optional[str]=None,
                 run_description:Optional[str]=None,
                 enable_wandb=False,
                 metrics_type='default',
                 log_dir:Optional[str]=None,
                 log_filename:Optional[str]=None,
                 summaries_filename:Optional[str]=None,
                 allow_overwrite_log=False,
                 summaries_stdout=True,
                 glabal_instance:Optional[bool]=None,
                 save_on_exit:bool=True,
                 ) -> None:

        global _logger, _except_handler_installed, \
            summary, log_config, info, warn, error, log_sys_info, shutdown, flush

        if glabal_instance!=False and _logger is None:
            _logger = self
            # module level methods to call global logging object
            summary = partial(Logger.summary, _logger)
            log_config = partial(Logger.log_config, _logger)
            info = partial(Logger.info, _logger)
            warn = partial(Logger.warn, _logger)
            error = partial(Logger.error, _logger)
            log_sys_info = partial(Logger.log_sys_info, _logger)
            shutdown = partial(Logger.shutdown, _logger)
            flush = partial(Logger.flush, _logger)

        self.has_shutdown = False
        self.start_time = timeit.default_timer()
        self._py_logger = None
        self._wandb_logger = None
        self.enable_wandb = enable_wandb
        self.master_process = master_process
        self.summaries_stdout = summaries_stdout
        self.log_filepath = None
        self.summaries_filepath = None
        self.quite_keys:Optional[Set[str]] = None
        self.summaries = {}

        if master_process:
            if log_dir:
                log_dir = utils.full_path(str(log_dir), create=True)
                if log_filename:
                    self.log_filepath = utils.full_path(os.path.join(log_dir, str(log_filename)))
                if summaries_filename:
                    self.summaries_filepath = utils.full_path(os.path.join(log_dir, str(summaries_filename)))

            self._py_logger = create_py_logger(filepath=self.log_filepath,
                                            allow_overwrite_log=allow_overwrite_log,
                                            project_name=project_name,
                                            run_name=run_name,
                                            run_description=run_description)

            if enable_wandb:
                if utils.is_debugging():
                    self._py_logger.warning('Wandb logging is disabled in debug mode.') # type: ignore
                else:
                    self._wandb_logger = create_wandb_logger(project_name, run_name,
                                                    std_metrics[metrics_type],
                                                    run_description)
            # else leave things to None

            if save_on_exit and _logger == self:
                install_atexit()

        if not _except_handler_installed:
            def handle_execpt(original_handler, logger:'Logger', exc_type, exc_value, exc_traceback):
                msg = utils.get_exception_str(exc_type, exc_value, exc_traceback)
                logger.error(msg)
                if original_handler is not None:
                    original_handler(exc_type, exc_value, exc_traceback)

            sys.excepthook = partial(handle_execpt, original_handler=sys.excepthook, logger=self)
            _except_handler_installed = True

    def log_config(self, config):
        if self.summaries_stdout and self._py_logger is not None:
            self._py_logger.info(_dict2msg({'project_config': config}))
        if self.enable_wandb and self._wandb_logger is not None:
            self._wandb_logger.config.update(config)

    def info(self, d:Union[str, Mapping[str,Any]], py_logger_only:bool=False):
        # if quite key is set and set is empoty then quite everything
        # if there are keys in quite then only allow messages with those keys
        if self.quite_keys is not None:
            if len(self.quite_keys)==0 or isinstance(d, str) or \
                    (isinstance(d, Mapping) and not self.quite_keys.intersection(d.keys())):
                return

        if self._py_logger is not None:
            msg = d # use separate variable to avoid changing d for wandb
            if isinstance(d, Mapping):
                msg = _dict2msg(d)
            self._py_logger.info(msg)

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
        self.summaries.update(d)

        if self.summaries_stdout and self._py_logger is not None:
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
                        'sys/python_version': f'{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}',
                        'sys/torch_version': torch.__version__,
                        'sys/is_anomaly_enabled': torch.is_anomaly_enabled(),
                        'sys/memory_gb': psutil.virtual_memory().available / (1024.0 ** 3),
                        'sys/cpu_count': psutil.cpu_count(),
                        'sys/cpu_freq': psutil.cpu_freq()._asdict(),
                        'sys/cpu_percent': psutil.cpu_percent(interval=1),
                        'sys/cpu_stats': psutil.cpu_stats()._asdict(),
                        'sys/cpu_count_logical': psutil.cpu_count(logical=True),
                        'sys/cpu_count_physical': psutil.cpu_count(logical=False),
                        'sys/cpu_name': os.popen('wmic cpu get name').read().strip().split('\n')[1] if os.name == 'nt' else os.popen('lscpu | grep "Model name"').read().strip().split(':')[1].strip(),
                        'sys/utils.free_disk_space': utils.free_disk_space(),

                        'env/RANK': os.environ.get('RANK', None),
                        'env/LOCAL_RANK': os.environ.get('LOCAL_RANK', None),
                        'env/WORLD_SIZE': os.environ.get('WORLD_SIZE', None),
                        'env/MASTER_ADDR': os.environ.get('MASTER_ADDR', None),
                        'env/MASTER_PORT': os.environ.get('MASTER_PORT', None),
                        'env/OMP_NUM_THREADS': os.environ.get('OMP_NUM_THREADS', None),
                        'env/CUDA_HOME': os.environ.get('CUDA_HOME', None),
                        'env/PYTORCH_CUDA_ALLOC_CONF': os.environ.get('PYTORCH_CUDA_ALLOC_CONF', None),
                        'env/CUDA_VISIBLE_DEVICES': os.environ.get('CUDA_VISIBLE_DEVICES', None),
                        'cuda/nccl_available': torch.distributed.is_nccl_available(), # type: ignore
                        'cuda/device_name': torch.cuda.get_device_name() if torch.cuda.is_available() else '<not_cuda>',
                        'cuda/device_count': torch.cuda.device_count(),
                        'cuda/cudnn.enabled': torch.backends.cudnn.enabled, # type: ignore
                        'cuda/cudnn.benchmark': torch.backends.cudnn.benchmark, # type: ignore
                        'cuda/cudnn.deterministic': torch.backends.cudnn.deterministic, # type: ignore
                        'cuda/cudnn.version': torch.backends.cudnn.version(), # type: ignore
                        'cuda/CUDA_VISIBLE_DEVICES': os.environ.get('CUDA_VISIBLE_DEVICES', None),
                        'dist/get_world_size': torch.distributed.get_world_size() if torch.distributed.is_initialized() else None, # type: ignore
                        'dist/get_rank': torch.distributed.get_rank() if torch.distributed.is_initialized() else None, # type: ignore
                        'dist/torch.distributed.is_initialized': torch.distributed.is_initialized(), # type: ignore
                        'dist/torch.distributed.is_available': torch.distributed.is_available(), # type: ignore
                        'dist/gloo_available': torch.distributed.is_gloo_available(), # type: ignore
                        'dist/mpi_available': torch.distributed.is_mpi_available(), # type: ignore

                        # TODO: importlib.metadata doesn't work in Python 3.8 so disabling for now
                        # 'flash_attn_ver': str(utils.get_package_ver('flash_attn')),
                        # 'transformers_ver': str(utils.get_package_ver('transformers')),
                        })

    def quite(self, except_keys:Optional[Union[str, Iterable[str]]]):
        if except_keys is not None:
            if not isinstance(except_keys, str):
                except_keys = set(list(except_keys))
            else:
                except_keys = set([except_keys])
        self.quite_keys = except_keys

    def shutdown(self, write_total_time:bool=True):
        if self.has_shutdown:
            return

        # log current exception
        try:
            if sys.exc_info()[0] is not None:
                exc_type, exc_value, traceback = sys.exc_info()
                self.error("Current Exception:", exception_instance=exc_value, stack_info=True)

                # Check for CUDA OOM error
                if isinstance(exc_value, torch.cuda.CudaError) and 'out of memory' in str(exc_value):
                    # Check if the platform is Linux
                    if not utils.is_windows():
                        cuda_summary = torch.cuda.memory_summary()
                        self.info(cuda_summary)
        except Exception as e:
            self.error("Exception occured while logging current exception:", exception_instance=e, stack_info=True)

        if write_total_time:
            self.summary({
                            'run/log_filepath': self.log_filepath,
                            'run/start_time': self.start_time,
                            'run/elapsed_hr': (timeit.default_timer() - self.start_time)/3600.0}
                         )

        # write summaries to file
        if self.master_process and self.summaries_filepath:
            with open(self.summaries_filepath, 'w', encoding='utf-8') as f:
                json.dump(self.summaries, f, indent=4)

        # attach artifacts to wandb
        if self.summaries_filepath:
            self.log_artifact(name='summaries_file', type='file', file_or_dir=self.summaries_filepath)
        if self.log_filepath:
            self.log_artifact(name='log_file', type='file', file_or_dir=self.log_filepath)

        # close loggers
        if self._wandb_logger is not None:
            self._wandb_logger.finish()
        if self._py_logger is not None:
            py_logging.shutdown()

        self.has_shutdown = True

    def flush(self):
        if self._py_logger is not None:
            for handler in self._py_logger.handlers:
                handler.flush()

