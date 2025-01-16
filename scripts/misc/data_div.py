# algorith to divide data among workers with minimum of imbalance

import math

def get_allocations(total, workers):
    start, end, remaining = 0, 0, total
    allocations = []
    while workers > 0:
        if not remaining:
            return None
        this_alloc = math.floor(remaining / workers)
        end += this_alloc
        remaining -= this_alloc
        allocations.append((start, end))
        start = end
        workers -= 1
    return allocations

data_len = 12
workers = 6

print(f'data_len: {data_len}')
print(f'workers: {workers}')

allocations = get_allocations(data_len, workers)
if allocations is None:
    print('Data cannot be divided among workers')
    exit()
for i, (start, end) in enumerate(allocations):
    print(f'Worker {i}: {end-start}, {start} - {end}')
