import numpy as np

class NASHGFunc:
    def __init__(self, n):
        self.n = n

    def func_value(self, x):
        if np.any(x < 0):
            return np.inf
        else:
            return 0
        
    def prox_opr(self, x):
        prox_x = np.maximum(0, x)
        return prox_x

    def prox_opr_coordinate(self, x, j):
        prox_x_j = np.maximum(0,x[j])
        return prox_x_j
        
    def prox_opr_block(self, x_block):
        prox_x_block = np.maximum(0,x_block)
        return prox_x_block