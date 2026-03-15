# tests/test_features.py

import numpy as np
from tda_detect.utils import BoundaryMatrix


def test_filled_triangle():
    d1 = BoundaryMatrix([[-1,0,1],[1,-1,0],[0,1,-1]])
    d2 = BoundaryMatrix([[1],[1],[1]])
    assert d1.betti(d2) == 0   # filled triangle has no loop

def test_empty_triangle():
    d1 = BoundaryMatrix([[-1,0,1],[1,-1,0],[0,1,-1]])
    d2 = BoundaryMatrix(np.zeros((3,0), dtype=int))
    assert d1.betti(d2) == 1   # empty triangle has one loop

def test_two_components():
    d1 = BoundaryMatrix([
        [-1,0,1,0,0,0],
        [1,-1,0,0,0,0],
        [0,1,-1,0,0,0],
        [0,0,0,-1,0,1],
        [0,0,0,1,-1,0],
        [0,0,0,0,1,-1],
    ])
    b0 = 6 - np.linalg.matrix_rank(d1.M)
    assert b0 == 2   # two separate triangles = two components