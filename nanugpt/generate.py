from typing import List, Optional, Mapping
import os

import torch

from nanugpt import utils
from nanugpt import common
from nanugpt.glogging import Logger

class Generator:
    def __init__(self, config:Mapping, logger:Logger) -> None:
        self.context_length = config['model']['module_kwargs']['context_length']

        self.device, self.amp_ctx, torch_info = common.setup_device(config, logger)

        checkpoint_dir = utils.full_path(config['generate']['checkpoint_path'])
        checkpoint_log_filepath = os.path.join(checkpoint_dir, 'checkpoint_log.yaml')
        # if file does not exist, try to find most recent checkpoint in out_dir
        if not os.path.exists(checkpoint_log_filepath):
            logger.warn(f"No checkpoint log found at {checkpoint_dir}, looking into latest subdirectory...")
            subdirs = [os.path.join(checkpoint_dir, d) for d in os.listdir(checkpoint_dir) if os.path.isdir(os.path.join(checkpoint_dir, d))]
            if len(subdirs) > 0:
                # sort subdirectories by modification time
                subdirs = sorted(subdirs, key=os.path.getmtime)
                checkpoint_log_filepath = None
                for subdir in reversed(subdirs):
                    if os.path.exists(os.path.join(subdir, 'checkpoint_log.yaml')):
                        checkpoint_dir = subdir
                        checkpoint_log_filepath = os.path.join(subdir, 'checkpoint_log.yaml')
                        break
                if checkpoint_log_filepath is None:
                    raise FileNotFoundError(f"No checkpoint log found in any subdirs at {checkpoint_dir}")
            else:
                raise FileNotFoundError(f"No checkpoints or subdirectories found in {checkpoint_dir}")
        else:
            checkpoint_dir = utils.full_path(checkpoint_dir)

        # log the checkpoint directory
        logger.info(f"Using checkpoint directory: {checkpoint_dir}")

        # check if checkpoint log exists
        if not os.path.exists(checkpoint_log_filepath):
            raise FileNotFoundError(f"No checkpoint log found at {checkpoint_log_filepath}")

        checkpoint_log = utils.load_yaml(checkpoint_log_filepath)
        if len(checkpoint_log) == 0:
            raise ValueError(f"No checkpoints found in {checkpoint_log_filepath}")

        # if checkpoint file name is not specified in config, use the most recent one
        checkpoint_filename = config['generate'].get('checkpoint_filepath', None)
        if checkpoint_filename:
            checkpoint_filepath = os.path.join(checkpoint_dir, utils.full_path(checkpoint_filename))
        else:
            checkpoint_filepath = checkpoint_log[-1]['checkpoint_filepath']

        logger.info(f"Using checkpoint file: {checkpoint_filepath}")

        checkpoint = torch.load(checkpoint_filepath,
                                map_location=self.device, weights_only=True)

        # load checkpoint
        state_dict = checkpoint['model']
        unwanted_prefix = '_orig_mod.'
        for k,v in list(state_dict.items()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)

        # create tokenizer
        self.tokenizer, tokenizer_config = common.create_tokenizer(config, logger)

        # create model
        self.model, model_config = common.create_model(config, logger, self.device,
                                                       vocab_size=len(self.tokenizer),
                                                       get_loss=None,
                                                       state_dict=state_dict)
        self.model.eval()

    @torch.no_grad()
    def generate(self, prompts:List[str], max_length:int, temperature:float=1.0, top_k:Optional[int]=None):
        if abs(temperature) < 1e-6:
            temperature = 1e-6 * (1 if temperature > 0 else -1)
        assert top_k is None or top_k > 0, f'top_k must be > 0, got {top_k}'

        self.model.eval()

        idxs = self.tokenizer.batch_encode(prompts)['input_ids']
        results = []
        for idx in idxs:
            idx = torch.tensor(idx, dtype=torch.long, device=self.device).unsqueeze(0)
            for gen_i in range(max_length):
                # if the sequence context is growing too long we must crop it at block_size
                idx_cond = idx if idx.size(1) <= self.context_length else idx[:, -self.context_length:]
                # forward the model to get the logits for the index in the sequence
                logits, *_ = self.model(idx_cond, only_last=True)
                # pluck the logits at the final step and scale by desired temperature
                logits = logits[:, -1, :] / temperature
                # optionally crop the logits to only the top k options
                if top_k is not None:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = -float('Inf')
                # apply softmax to convert logits to (normalized) probabilities
                probs = torch.nn.functional.softmax(logits, dim=-1)
                # sample from the distribution
                idx_next = torch.multinomial(probs, num_samples=1)
                # append sampled index to the running sequence and continue
                idx = torch.cat((idx, idx_next), dim=1)
            results.append(idx.squeeze(0).tolist())
        return self.tokenizer.batch_decode(results)
