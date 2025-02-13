import torch
from torch.utils.data import DataLoader
from nanugpt.data.arithmatic_data import get_data

def test_data_loaders():
    device_batch_size = 10
    eval_batch_size = 10
    data_loader_seed = 42
    min_digits = 1
    max_digits = 2
    max_samples = 50
    context_len = 20

    train_loader, val_loader, test_loader = get_data(device_batch_size, eval_batch_size,
                                                     data_loader_seed, min_digits,
                                                     max_digits, max_samples, context_len)

    # Check loaders are DataLoader instances
    assert isinstance(train_loader, DataLoader)
    assert isinstance(val_loader, DataLoader)

    # Test one batch from train_loader
    for data, target in train_loader:
        # Each sequence should have the specified context length
        assert data.size(1) == context_len
        assert target.size() == data.size()
        # Verify shifting: target[:, :-1] should equal data[:, 1:]
        assert torch.equal(data[:, 1:], target[:, :-1])
        break

    # Test one batch from val_loader
    for data, target in val_loader:
        assert data.size(1) == context_len
        break

    print("Data loaders test passed.")

if __name__ == '__main__':
    test_data_loaders()
