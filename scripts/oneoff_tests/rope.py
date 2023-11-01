import torch
import numpy as np

def get_rotary_matrix(context_window:int, embedding_dim:int):
    R = torch.zeros((context_window, embedding_dim, embedding_dim), requires_grad=False)
    for position in range(context_window):
        for i in range(embedding_dim//2):
            theta = 10000. ** (-2.*(i - 1) / embedding_dim)
            m_theta = position * theta
            R[position, 2*i,2*i] = np.cos(m_theta)
            R[position, 2*i,2*i+1] = - np.sin(m_theta)
            R[position, 2*i+1,2*i] = np.sin(m_theta)
            R[position, 2*i+1,2*i+1] = np.cos(m_theta)
    return R

d_model = 2
context_window = 3

assert d_model % 2 == 0

R = get_rotary_matrix(context_window, d_model) # (context_window, d_model, d_model)
x = torch.randn(d_model)
y = torch.randn(d_model)

print('R')
print(R)
print('-----------')
print('x')
print(x)
print('-----------')
print('y')
print(y)

m = 0
n = 1


# if you take inner product of rotated version of x and y at pos m and n,
# i.e., x_m and y_n,
# the result is same as if x was at pos 0 and y was at pos m-n
ans1 = (R[m,:,:] @ x) @ (R[n,:,:] @ y) # scaler
ans2 = (x @ R[n-m,:,:]) @ y # scaler
ans3 = (R[m+1,:,:] @ x) @ (R[n+1,:,:] @ y)

print(ans1, ans2, ans3)