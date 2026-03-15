import numpy as np
import ripser
from persim import plot_diagrams
from numpy.linalg import matrix_rank

# Cell 2
d1 = np.array([
    [-1,   0,  +1],  
    [+1,  -1,   0],  
    [ 0,  +1,  -1],  
], dtype=float)

d2 = np.array([
    [+1],  
    [+1],  
    [+1],  
], dtype=float)

print('∂1 @ ∂2 =\n', d1 @ d2)

# Cell 3
def betti(d_k, d_k1):
    n_cols  = d_k.shape[1]
    rank_dk = matrix_rank(d_k)
    ker_dim = n_cols - rank_dk
    im_dim  = matrix_rank(d_k1)
    return ker_dim - im_dim

b0 = 3 - matrix_rank(d1)
b1 = betti(d1, d2)
print(f'β₀ = {b0}')
print(f'β₁ = {b1}')

d2_empty = np.zeros((3, 0))
b1_empty = betti(d1, d2_empty)
print(f'\nEmpty triangle: β₁ = {b1_empty}')

# Cell 4
theta = np.linspace(0, 2*np.pi, 80)
X = np.column_stack([np.cos(theta), np.sin(theta)])
X += np.random.normal(0, 0.05, X.shape)

try:
    result = ripser.ripser(X, maxdim=1)
    print("ripser success")
except Exception as e:
    print(f"ripser error: {e}")
