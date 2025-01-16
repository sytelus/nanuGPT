import math

n_tokens = 12
context_length = 2
world_size = 4

n_context = n_tokens / context_length
n_context_per_rank = n_context / world_size

print(f'Total tokens: {n_tokens}')
print(f'Context length: {context_length}')
print(f'World size: {world_size}')
print(f'Total context: {n_context}')
print(f'Context per rank: {n_context_per_rank}')

for global_rank in range(world_size):
    rank_start = int(n_context * float(global_rank) / world_size) * context_length
    rank_end = rank_start + math.ceil(n_context_per_rank) * context_length
    print(f'Rank {global_rank}: {rank_end-rank_start} , {rank_start} - {rank_end-1}')