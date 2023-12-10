from typing import Optional, Mapping, Tuple
import os
import sys
import dataclasses
from contextlib import AbstractContextManager, nullcontext
from packaging import version
import json
import atexit

import torch

from nanugpt import utils
from nanugpt import glogging as logging
from nanugpt.tokenizers.tokenizer_base import TokenizerBase

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

    d = {'torch_info/'+k:v for k,v in dataclasses.asdict(torch_info).items()}
    d['torch_info/pt_dtype'] = str(d['torch_info/pt_dtype'])  # make it JSON serializable so it can be logged
    logger.summary(d)

    device = torch.device(torch_info.device_name)
    amp_ctx = nullcontext() if torch_info.device_type == 'cpu' else torch.amp.autocast(device_type=torch_info.device_type, dtype=torch_info.pt_dtype)

    return device, amp_ctx, torch_info

def setup_logger(is_master:bool,config:Optional[Mapping]=None, logger:Optional[logging.Logger]=None)->logging.Logger:
    if logger is None:
        print(f"Creating logger on LOCAL_RANK: {os.environ.get('LOCAL_RANK', '-1')}, RANK {os.environ.get('RANK', '-1')}, is_master={is_master}, utils.is_master_process={utils.is_master_process()}")
        assert config is not None, "Either config or logger must be provided but both are None."
        logging_config = config['logging']
        if not logging_config['log_dir']:
            out_dir = config['data']['tokenized_out_dir']
            logging_config['log_dir'] = utils.full_path(out_dir, create=True)
        logger = logging.Logger(master_process=is_master, **logging_config)

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
                           vocab_size:int, state_dict=None)->Tuple[torch.nn.Module, Mapping]:
    model_config = config['model']
    torch_compile = config['general']['torch_compile']

    get_model = utils.import_fn(model_config['module'])

    model = get_model(vocab_size=vocab_size,
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



