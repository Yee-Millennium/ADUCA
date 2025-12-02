import numpy as np

class GMVIProblem:
    def __init__(self, operator_func, g_func):
        self.n = operator_func.n
        self.operator_func = operator_func
        self.g_func = g_func
    
    def residual(self, q):
        return np.linalg.norm(q - self.g_func.prox_opr(q - self.operator_func.func_map(q)))