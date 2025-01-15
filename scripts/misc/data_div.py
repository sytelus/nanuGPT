# algorith to divide data among workers with minimum of imbalance

import math

data_len = 1009
workers = 7

print(f'data_len: {data_len}')
print(f'workers: {workers}')

allocated = 0
next_start = 0
i = 0
while workers > 0:
    this_alloc = math.floor(data_len / workers)
    allocated += this_alloc
    print(f'worker {i}: {this_alloc} for {next_start}-{allocated-1}')
    next_start = allocated
    data_len -= this_alloc
    workers -= 1
    i += 1

print(f'allocated: {allocated}')
print(f'next_start: {next_start}')
