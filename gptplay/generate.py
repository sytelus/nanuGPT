import math
from typing import List, Optional, Mapping, Tuple
import os

import torch

from gptplay import utils
from gptplay.config import Config
from gptplay import common
from gptplay import logging

class Generator:
    def __init__(self, config:Mapping, logger) -> None:
        self.context_length = config['model']['module_kwargs']['context_length']

        self.device, self.amp_ctx, self.logger, torch_info = common.setup_device(config, logger)

        # load checkpoint log
        checkpoint_log_filepath = os.path.join(config['general']['out_dir'], 'checkpoint_log.yaml')
        checkpoint_log = utils.load_yaml(checkpoint_log_filepath)
        if len(checkpoint_log) == 0:
            raise ValueError(f"No checkpoints found in {checkpoint_log_filepath}")

        checkpoint_filepath = checkpoint_log[-1]['checkpoint_filepath']
        checkpoint = torch.load(checkpoint_filepath, map_location=self.device)

        # load checkpoint
        state_dict = checkpoint['model']
        unwanted_prefix = '_orig_mod.'
        for k,v in list(state_dict.items()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)

        # load model and tokenizer
        self.model, self.tokenizer = common.create_model_tokenizer(config, logger, self.device, state_dict=state_dict)
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
                logits = self.model(idx_cond, only_last=True)
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

if __name__ == "__main__":
    # specify config file to use as first argument in commandline
    config = Config(default_config_filepath='configs/train_llm/tinyshakespeare.yaml')
    logging_config = config['logging']
    logger = logging.Logger(master_process=True, **logging_config)

    gen = Generator(config, logger)
    results = gen.generate(['\n'], 200)
    print(results)

    logger.all_done()