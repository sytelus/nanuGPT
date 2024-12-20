import numpy as np
from tqdm.auto import tqdm

from nanugpt import common, utils
from nanugpt.config import Config
from nanugpt import glogging as logging

def analyze_documents(file_path, eos_token_id, dtype):
    file_path = utils.full_path(file_path)

    # Open the memory-mapped file
    mmapped_file = np.memmap(file_path, dtype=dtype, mode='r')

    file_length = len(mmapped_file)
    chunk_size = min(file_length, 1024 * 1024 * 1024)  # 1 GB
    document_lengths = []
    eos_indices = np.array([], dtype=np.int64)

    for chunk_start in tqdm(range(0, file_length, chunk_size),
                            total=file_length//chunk_size,
                            desc="Analyzing documents"):
        chunk_end = min(chunk_start + chunk_size, file_length)
        chunk = mmapped_file[chunk_start:chunk_end]

        # Find indices of EOS tokens in the chunk
        eos_indices = np.append(eos_indices,
                                np.where(chunk == eos_token_id)[0] + chunk_start)

    # Cleanup
    del mmapped_file

    # before diff add 0 for first document
    eos_indices = np.insert(eos_indices, 0, 0)

    # if there was no EOS at end, add one
    if len(eos_indices)<2 or eos_indices[-1] != file_length - 1:
        eos_indices = np.append(eos_indices, file_length - 1)

    document_lengths = np.diff(eos_indices) + 1

    # Calculating statistics
    total_tokens = np.sum(document_lengths).item()
    num_documents = len(document_lengths)
    avg_tokens_per_doc = np.mean(document_lengths).item()
    min_tokens_per_doc = np.min(document_lengths).item()
    max_tokens_per_doc = np.max(document_lengths).item()
    stddev_tokens_per_doc = np.std(document_lengths).item()
    median_tokens_per_doc = np.median(document_lengths).item()
    upper95_percentile = np.percentile(document_lengths, 95).item()
    coverage_1k_ctx = np.sum(document_lengths <= 2**10)*100.0 / len(document_lengths)
    coverage_2k_ctx = np.sum(document_lengths <= 2**11)*100.0 / len(document_lengths)
    coverage_4k_ctx = np.sum(document_lengths <= 2**12)*100.0 / len(document_lengths)
    coverage_8k_ctx = np.sum(document_lengths <= 2**13)*100.0 / len(document_lengths)

    result = {
        'total_tokens': total_tokens,
        'num_documents': num_documents,
        'avg_tokens_per_doc': avg_tokens_per_doc,
        'min_tokens_per_doc': min_tokens_per_doc,
        'max_tokens_per_doc': max_tokens_per_doc,
        'stddev_tokens_per_doc': stddev_tokens_per_doc,
        'median_tokens_per_doc': median_tokens_per_doc,
        'upper95_percentile': upper95_percentile,
        'coverage_1k_ctx': coverage_1k_ctx,
        'coverage_2k_ctx': coverage_2k_ctx,
        'coverage_4k_ctx': coverage_4k_ctx,
        'coverage_8k_ctx': coverage_8k_ctx,
    }

    logging.summary(result)

    print(utils.dict2tsv(result, delimiter=';'))


if __name__ == "__main__":
    # specify config file to use as first argument in commandline
    config = Config(default_config_filepath='configs/train_gpt2/wikitext103.yaml')
    config['logging']['log_filename'] = 'tokens_analysis.log'
    config['logging']['enable_wandb'] = False
    config['logging']['project_name'] = 'tokens_analysis'
    config['logging']['run_name'] = 'tokens_analysis'

    logger = common.setup_logger(config=config)

    tokenized_train_path = config['data']['module_kwargs']['tokenized_train_path']
    dtype = config['data']['module_kwargs']['dtype']

    # create tokenizer
    tokenizer, tokenizer_config = common.create_tokenizer(config, logger)
    logger.summary({'run/vocab_size': len(tokenizer)})

    analyze_documents(tokenized_train_path, tokenizer.eot_token_id(), dtype)