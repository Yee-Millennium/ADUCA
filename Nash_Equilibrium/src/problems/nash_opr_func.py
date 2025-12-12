import numpy as np
import warnings
# from scipy.linalg.blas import dgemm
from scipy.sparse import csr_matrix

class NASHOprFunc:
    def __init__(self, n, gamma, beta, c , L):
        self.n = n
        self.gamma = gamma
        self.beta = beta
        self.c = c
        self.L = L
        # Prevent zero/negative totals from hitting invalid power operations
        self._min_total_quantity = 1e-12

    def _clip_total_quantity(self, Q):
        return max(Q, self._min_total_quantity)

    def f(self, q):
        q = np.maximum(q, 0)
        t = 1/self.beta
        res = self.c * q + 1/(1+t)*(self.L**t * q**(1+t))
        return res
    
    def f_block(self, q, block:range):
        q_block = np.maximum(q[block], 0)
        t = 1./self.beta[block]
        res = self.c[block] * q_block + 1./(1.+t[block])*(self.L[block]**t * q_block**(1+t))
        return res

    def df(self, q):
        q = np.maximum(q, 0)
        t = 1./self.beta
        res = self.c + (self.L* q)**t
        return res
    
    def df_block(self, q_block, block:range):
        q_block = np.maximum(q_block, 0)
        t = 1./self.beta[block]
        res = self.c[block] + (self.L[block]* q_block)**t
        return res
    
    def p(self, Q):
        Q = self._clip_total_quantity(Q)
        return (5000**(1/self.gamma)) * (Q**(-1/self.gamma))
        
    
    def dp(self, Q):
        Q = self._clip_total_quantity(Q)
        res = -1./self.gamma * (5000**(1./self.gamma)) * (Q**(-1/self.gamma - 1))
        return res
    
    def func_map(self, q):
        Q = np.sum(q)
        res = self.df(q) - self.p(Q) - q*self.dp(Q)
        return res

    def func_map_block(self, q_block, Q, block):
        res = self.df_block(q_block, block) - self.p(Q) - q_block*self.dp(Q)
        return res
    
    def func_map_block_update(self, F, q, p, p_, dp, dp_, block:range):
        F[block] = self.df_block(q[block], block) - p - q[block] * dp
        F[:block.start] += (p_ - p) + q[:block.start]*(dp_ - dp)
        F[block.stop:] += (p_ - p) + q[block.stop:]*(dp_ - dp)
        return F

    # def func_map_block_sample(self, j, t, x):
    #     assert len(x) == self.d + self.n
    #     assert 1 <= j <= self.d + self.n
    #     assert 1 <= t <= self.n

    #     if j <= self.d:
    #         return x[self.d + t - 1] * self.b[t - 1] * self.A[t - 1, j - 1]
    #     elif j - self.d == t:
    #         return - (self.b[t - 1] * (self.A[t - 1, :] @ x[:self.d]) - 1)
    #     else:
    #         return 0.0
