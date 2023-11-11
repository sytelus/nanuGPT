from typing import Optional, Mapping, Tuple
import os
import sys
import dataclasses
from contextlib import AbstractContextManager, nullcontext
from packaging import version

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

    torch_info = utils.setup_torch(seed=seed,
                device_type=device_type, dtype=dtype,
                enable_distributed=enable_distributed,
                distributed_backend=distributed_backend,
                distributed_init_method=distributed_init_method)

    utils.setup_sys(seed + torch_info.seed_offset)

    d = dataclasses.asdict(torch_info)
    d['pt_dtype'] = str(d['pt_dtype'])  # make it JSON serializable so it can be logged
    logger.summary(d)

    device = torch.device(torch_info.device_name)
    amp_ctx = nullcontext() if torch_info.device_type == 'cpu' else torch.amp.autocast(device_type=torch_info.device_type, dtype=torch_info.pt_dtype)

    return device, amp_ctx, torch_info

def setup_logger(is_master:bool,config:Optional[Mapping]=None, logger:Optional[logging.Logger]=None)->logging.Logger:
    if logger is None:
        print(f"Creating logger on LOCAL_RANK: {os.environ.get('LOCAL_RANK', '-1')}, RANK {os.environ.get('RANK', '-1')}")
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
            model = torch.compile(model) # requires PyTorch 2.0
        except Exception as e:
            logger.error(f"Failed to compile model: {str(e)}")
        logger.info("Compiling done.")

    return model


def create_model_tokenizer(config:Mapping, logger:logging.Logger, device:torch.device,
                           state_dict=None)->Tuple[torch.nn.Module, TokenizerBase, Mapping, Mapping]:
    model_config = config['model']
    tokenizer_config = config['tokenizer']
    torch_compile = config['general']['torch_compile']

    get_tokenizer_factory = utils.import_fn(tokenizer_config['module'])
    get_model = utils.import_fn(model_config['module'])

    tokenizer_factory = get_tokenizer_factory(**tokenizer_config['module_kwargs'])
    tokenizer = tokenizer_factory()

    model = get_model(vocab_size=len(tokenizer),
                      **model_config['module_kwargs']).to(device)
    logger.summary({'model_params_all': utils.module_params_count(model, non_embedding=False),
                    'model_params_no_embedding': utils.module_params_count(model, non_embedding=True),})

    if state_dict is not None:
        logger.info("Loading model from state_dict...")
        model.load_state_dict(state_dict)

    if torch_compile:
        model = compile_torch_model(model, logger)

    return model, tokenizer, model_config, tokenizer_config

def check_env_vars():
    utils.set_env_vars({'OUT_DIR': ('output', None),
                        'DATA_ROOT': (None, 'This variable should be set to directory where you data resides')
                       }, raise_exec=True)
