from typing import Optional, Tuple
import torch

from torch.utils.data import DataLoader

from nanugpt.tokenizers.grokking_tokenizer import GrokkingTokenizer, get_tokenizer_factory, DIVISION_MODULO_OPERATIONS, ALL_OPERATIONS

# Implements get_data interface for Grokking dataset

def operation_mod_p_data(operation: str, p: int, tokenizer: GrokkingTokenizer):
    """
    a◦b (mod p) for 0 <= a < p, 1 <= b < p if operation in DIVISION_MODULO_OPERATIONS
    a◦b (mod p) for 0 <= a, b < p otherwise

    NOTE: This code is written specifically to be memory efficient for large vocab
    """
    eos = tokenizer[tokenizer.eos_token]
    eq = tokenizer[tokenizer.eq_token]
    op = tokenizer[operation]

    equations = torch.cartesian_prod(torch.arange(p),
                                     torch.arange(0 if not operation in DIVISION_MODULO_OPERATIONS else 1, p))

    # shuffle combinations
    # # Generate a permutation of row indices
    # permuted_indices = torch.randperm(equations.size(0))
    # # Shuffle the rows using the permuted indices
    # equations = equations[permuted_indices]

    equations = ALL_OPERATIONS[operation](equations[:,0], equations[:,1], p) # tuple of 3 tensors
    equations = torch.stack((equations[0], equations[1], equations[2] % p), dim=1) # turn result column into modulo p

    # map tokens to ids
    token_to_idx = torch.tensor(list(tokenizer[i] for i in range(torch.max(equations)+1)), dtype=torch.int32)
    equations = token_to_idx[equations]

    # 3rd column is the result, must be LongTensor for classification
    results = equations[:,2].to(torch.int64)

    # first two columns are the inputs
    equations = equations[:,:2]

    # insert columns for eos, op, eq
    # Create tensors for k1 and k2 with the same number of rows as the original tensor
    eos_t = torch.full((equations.size(0), 1), eos)
    op_t = torch.full((equations.size(0), 1), op)
    eq_t = torch.full((equations.size(0), 1), eq)
    equations = torch.cat((eos_t, equations[:, :1], op_t, equations[:, 1:], eq_t), dim=1)

    return equations, results

def get_data(operation: str, prime: int, training_fraction: float, val_fraction:Optional[float],
             device_batch_size: int, eval_batch_size:int, data_loader_seed:int,
             local_rank:int, context_length:int)->Tuple[DataLoader,DataLoader, Optional[DataLoader]]:
    tokenizer = get_tokenizer_factory(prime)()

    inputs, labels = operation_mod_p_data(operation, prime, tokenizer)

    dataset = torch.utils.data.TensorDataset(inputs, labels)

    train_size = int(training_fraction * len(dataset))
    if val_fraction:
        val_size = int(val_fraction * len(dataset))
        test_size = len(dataset) - train_size - val_size
    else:
        val_size = len(dataset) - train_size
        test_size = 0

    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset,
                                                               [train_size, val_size, test_size])

    train_loader_seed, val_loader_seed, test_loader_seed = data_loader_seed+local_rank, data_loader_seed+local_rank + 1, data_loader_seed+local_rank + 2

    train_loader_gen = torch.Generator().manual_seed(train_loader_seed)
    val_loader_gen = torch.Generator().manual_seed(val_loader_seed)

    train_loader = DataLoader(train_dataset,
                              batch_size=min(device_batch_size, len(train_dataset)) ,
                              shuffle=True,
                              num_workers=1, # don't use main process as worker
                              generator=train_loader_gen)
    val_loader = DataLoader(val_dataset,
                            batch_size=min(eval_batch_size, len(val_dataset)) ,
                            shuffle=False,
                            num_workers=1, # don't use main process as worker
                            generator=val_loader_gen)

    if len(test_dataset):
        test_loader_gen = torch.Generator().manual_seed(test_loader_seed)
        test_loader = DataLoader(test_dataset,
                                 batch_size=min(eval_batch_size, len(test_dataset)) ,
                                 shuffle=False,
                                 num_workers=1, # don't use main process as worker
                                 generator=test_loader_gen)
    else:
        test_loader = None

    return train_loader, val_loader, test_loader
