from typing import Optional, Tuple, List, Dict, Mapping


def get_data(operation: str, prime: int, training_fraction: float, val_fraction:Optional[float],
             train_batch_size: int, eval_batch_size:int, data_loader_seed:int,
             local_rank:int, context_length:int)->Tuple[DataLoader,DataLoader, Optional[DataLoader]]:
    pass