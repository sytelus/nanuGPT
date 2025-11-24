
from typing import Optional, Tuple

from archai.nlp.tokenizer_utils.vocab_base import VocabBase
from archai.nlp.tokenizer_utils.word_vocab import WordVocab
from archai.nlp.tokenizer_utils.bbpe_vocab import BbpeVocab
from archai.nlp.tokenizer_utils.gpt2_vocab import Gpt2Vocab

def _create_vocab(datadir:str, dataset:str, vocab_type:str, vocab_cache_dir:str,
                    vocab_size:Optional[int]=None)->VocabBase:
    if vocab_type == 'word':
        # '<S>' is added for double eos and <unk> is rare token in corpus with freq < 3
        bos_token, eos_token, lower_case, vocab_file = None, '<eos>', False, None # vocab file is text file of symbols, one per line
        if dataset in ['wt103', 'wt2', 'olx']:
            pass
        elif dataset == 'ptb':
            lower_case = True
        elif dataset == 'lm1b':
            bos_token, eos_token, vocab_file = '<S>', '<S>', os.path.join(datadir, '1b_word_vocab.txt')
        elif dataset in ['enwik8', 'text8']:
            eos_token, lower_case = None, True
        else:
            raise RuntimeError(f'dataset {dataset} is not recognized to produce vocab')

        vocab = WordVocab(save_path=vocab_cache_dir, vocab_size=vocab_size,
                            bos_token=bos_token, eos_token=eos_token,
                            lower_case=lower_case)
    elif vocab_type == 'bbpe':
        vocab = BbpeVocab(save_path=vocab_cache_dir, vocab_size=vocab_size or 50257) # default vocab size for GPT-2 is 50257
    elif vocab_type == 'gpt2':
        vocab = Gpt2Vocab(save_path=vocab_cache_dir, vocab_size=vocab_size or 50257) # default vocab size for GPT-2 is 50257
    else:
        raise RuntimeError(f'Unsupported vocab type: {vocab_type}')

    return vocab


def _train_vocab(self)->None:
    if self.refresh_cache or not self.vocab.is_trained(): # if vocab cache does not exist
        train_filepath, valid_filepath, test_filepath = \
            self._dataset_filepaths()

        logging.info('Training vocab...')
        self.vocab.train([train_filepath])
        logging.info('Finished training vocab.')
    else:
        self.vocab.load()
        logging.info(f'Vocab cache found and loaded for type {self.vocab_type} and size {self.vocab_size} from {self._vocab_cache_dir}.')


def _create_train_vocab(self)->VocabBase:
    vocab = _create_vocab(self.datadir, self.dataset, self.vocab_type,
                                    self._vocab_cache_dir, vocab_size=self.vocab_size)
    _train_vocab()

    return self.vocab


vocab = _create_train_vocab()