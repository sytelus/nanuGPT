from typing import Optional, Mapping, Tuple, Callable, Union, TypeAlias
import os
import sys
import dataclasses
from contextlib import AbstractContextManager, nullcontext
from packaging import version
import json

import torch

from nanugpt import utils
from nanugpt import glogging as logging
from nanugpt.tokenizers.tokenizer_base import TokenizerBase

GetLossType:TypeAlias = Callable[[Union[torch.Tensor, Mapping], torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]

def setup_device(config:Mapping, logger:logging.Logger)->Tuple[torch.device, AbstractContextManager, utils.TorchInfo]:
    seed = config['general']['seed']
    device_type = config['general']['device_type']
    dtype = config['general']['dtype']
    enable_distributed = config['general']['enable_distributed']
    distributed_backend = config['general']['distributed_backend']
    distributed_init_method = config['general']['distributed_init_method']

    if enable_distributed is None and int(os.environ.get('WORLD_SIZE', '1')) > 1:
        enable_distributed = True

    # load cuda modules on demand to save memory
    # TODO: add config for below, should be false by default
    # if 'CUDA_MODULE_LOADING' not in os.environ:
    #     os.environ['CUDA_MODULE_LOADING'] = 'LAZY'

    torch_info = utils.setup_torch(seed=seed,
                device_type=device_type, dtype=dtype,
                enable_distributed=enable_distributed,
                distributed_backend=distributed_backend,
                distributed_init_method=distributed_init_method)

    utils.setup_sys(seed + torch_info.seed_offset)

    device = torch.device(torch_info.device_name)
    amp_ctx = nullcontext() if torch_info.device_type == 'cpu' else torch.amp.autocast(device_type=torch_info.device_type, dtype=torch_info.pt_dtype)

    d = {'torch_info/'+k:v for k,v in dataclasses.asdict(torch_info).items()}
    d['torch_info/pt_dtype'] = str(d['torch_info/pt_dtype'])  # make it JSON serializable so it can be logged
    d['torch_info/device_index'] = str(device.index)  # make it JSON serializable so it can be logged
    logger.summary(d)
    logger.log_torch_info()

    return device, amp_ctx, torch_info

def setup_logger(config:Optional[Mapping]=None, logger:Optional[logging.Logger]=None)->logging.Logger:
    if logger is None:
        print(f"Creating logger on LOCAL_RANK: {os.environ.get('LOCAL_RANK', '-1')}, GLOBAL_RANK {utils.get_global_rank()}, IS_MASTER={utils.is_master_process()}")

        # before we create logger, let's print full command line we received
        print(f"[{utils.get_global_rank()}] command line: {' '.join(sys.argv)}", flush=True)

        check_env_vars()

        assert config is not None, "Either config or logger must be provided but both are None."
        logging_config = config['logging']
        if not logging_config['log_dir']:
            out_dir = config['data']['tokenized_out_dir']
            logging_config['log_dir'] = utils.full_path(out_dir, create=True)
        logger = logging.Logger(**logging_config)

    logger.log_sys_info()
    logger.log_config(config)

    return logger


def compile_torch_model(model:torch.nn.Module, logger:logging.Logger)->torch.nn.Module:
    python_version = sys.version_info
    pytorch_version = version.parse(torch.__version__)
    if python_version >= (3,11) and pytorch_version < version.parse('2.1.0'):
        logger.warn(f"PyTorch {pytorch_version} does not support Python {python_version} for model compilation.")
    elif utils.is_windows() and pytorch_version <= version.parse('2.2.0'):
        logger.warn(f"PyTorch {pytorch_version} does not support Windows for model compilation.")
    else:
        logger.info("Compiling model...")
        try:
            #torch._dynamo.config.verbose=True # if compile error outs
            model = torch.compile(model) # type: ignore
        except Exception as e:
            logger.error(f"Failed to compile model: {str(e)}")
        logger.info("Compiling done.")

    return model


def create_model(config:Mapping, logger:logging.Logger, device:torch.device,
                vocab_size:int, get_loss:Optional[GetLossType],
                state_dict=None)->Tuple[torch.nn.Module, Mapping]:
    model_config = config['model']
    torch_compile = config['general']['torch_compile']

    get_model = utils.import_fn(model_config['module'])

    model = get_model(vocab_size=vocab_size, get_loss=get_loss,
                      **model_config['module_kwargs']).to(device)

    if state_dict is not None:
        logger.info("Loading model from state_dict...")
        model.load_state_dict(state_dict)

    if torch_compile:
        model = compile_torch_model(model, logger)

    return model, model_config

def create_tokenizer(config:Mapping, logger:logging.Logger)->Tuple[TokenizerBase, Mapping]:
    tokenizer_config = config['tokenizer']
    get_tokenizer_factory = utils.import_fn(tokenizer_config['module'])
    tokenizer_factory = get_tokenizer_factory(**tokenizer_config['module_kwargs'])
    tokenizer = tokenizer_factory()

    return tokenizer, tokenizer_config

def check_env_vars():
    utils.set_env_vars({'OUT_DIR': ('output', None),
                        'DATA_ROOT': (None, 'This variable should be set to directory where you data resides')
                       }, raise_exec=True)

def get_model_sizes()->Mapping[str, int]:
    with open(utils.full_path('nanugpt/assets/model_sizes.json'), mode='r', encoding='utf-8') as f:
        d = json.load(f)
    return d

def save_artifacts(out_dir:str, config:Mapping, logger:logging.Logger)->None:
    # save ccode dir
    logger.log_artifact(name='code', type='code', file_or_dir=utils.get_code_dir(),
                                desc_markdown='Training code directory')

    # save env vars
    # write all env vars to env.yaml in out_dir
    env_filepath = os.path.join(out_dir, "env.yaml")
    utils.save_env_vars(env_filepath)
    logger.log_artifact(name='env', type='yaml', file_or_dir=env_filepath,
                        desc_markdown="Environment variables at the start of the run")

    # attach config as artifact
    config_filepath = os.path.join(out_dir, "config_saved.yaml")
    utils.save_yaml(config, config_filepath)
    logger.log_artifact(name='config', type='yaml', file_or_dir=config_filepath,
                        desc_markdown="Configuration file at the start of the run")

    # save command line in shell script and attach as artifact
    cmd_filepath = os.path.join(out_dir, "command_line.sh")
    with open(cmd_filepath, 'w', encoding='utf-8') as f:
        f.write("#!/bin/bash\n")
        f.write(f"# Command line: {' '.join(utils.get_command_line())}\n")
        f.write(f"# Working directory: {os.getcwd()}\n")
        f.write("\n")
        f.write(' '.join(utils.get_command_line()) + '\n')
    os.chmod(cmd_filepath, 0o755) # make it executable
    logger.log_artifact(name='command_line', type='script', file_or_dir=cmd_filepath,
                        desc_markdown="Command line used to start the run")
