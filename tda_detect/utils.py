import numpy as np
from numpy.linalg import matrix_rank


class BoundaryMatrix:
    """
    Boundary operator ∂_k over Z/2Z.
    Rows = (k-1)-simplices, Columns = k-simplices.
    """

    def __init__(self, matrix):
        self.M = np.array(matrix, dtype=int)   # store raw, no mod 2 here

    def reduce(self):
        M = (self.M % 2)                       # convert to Z/2Z only for reduction
        n_cols = M.shape[1]
        pivot_col = {}

        for j in range(n_cols):
            rows_with_1 = np.where(M[:, j] == 1)[0]

            while len(rows_with_1) > 0:
                low = rows_with_1[-1]
                if low not in pivot_col:
                    pivot_col[low] = j
                    break
                else:
                    M[:, j] = (M[:, j] + M[:, pivot_col[low]]) % 2
                    rows_with_1 = np.where(M[:, j] == 1)[0]

        self.reduced = M
        self.pivot_col = pivot_col
        return M

    def betti(self, higher_boundary=None):
        n_cols  = self.M.shape[1]
        rank_dk = matrix_rank(self.M)
        ker_dim = n_cols - rank_dk

        if higher_boundary is None or higher_boundary.M.shape[1] == 0:
            im_dim = 0
        else:
            im_dim = matrix_rank(higher_boundary.M)

        return ker_dim - im_dim