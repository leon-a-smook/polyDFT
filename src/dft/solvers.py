# dft/solvers.py
import numpy as np
from scipy.sparse.linalg import LinearOperator, gmres
from .core import get_Crank_Nicolson_mat, propagate_q, get_rho_p

class Anderson:
    def __init__(self, m=5, beta=0.3, adaptive_beta = True, alpha=0.01, pre_iters=10):
        self.m = m                  # history depth
        self.beta = beta            # mixing damping
        self.adaptive_beta = adaptive_beta
        self.iters = 0              # Track iterations
        self.pre_iters = pre_iters
        self.alpha = alpha
        self.reset()
    
    def reset(self):
        self.res_hist = []          # residuals: r_k = w_new - w
        self.field_hist = []        # field differences: dw(k) = w(k) - w(k-1)
        self.w_prev = None          # stores previous w_k
        self.prev_res_norm = None   # residuals norm previous step

    def mix(self, w, w_new):
        self.iters += 1              # Add counter to iterations so all exits are counted
        r_k = w_new - w              # newest residuals

        if self.w_prev is not None:
            f_k = w - self.w_prev
            self.field_hist.append(f_k.copy())
            self.res_hist.append(r_k.copy())

            # remove old entries to maintain history size
            if len(self.res_hist) >= self.m:
                self.res_hist.pop(0)
                self.field_hist.pop(0)

        self.w_prev = w.copy()

        if self.iters < self.pre_iters:
            return (1-self.alpha)*w + self.alpha*w_new

        res_norm = np.linalg.norm(r_k)
        if self.prev_res_norm is not None and res_norm > self.prev_res_norm * 1.05:
            return w + self.beta * r_k  # fallback to simple mixing
        self.prev_res_norm = res_norm

        if self.adaptive_beta:
            beta = self.beta / (1 + res_norm)
        else:
            beta = self.beta

        if len(self.res_hist) < 1:
            return w + self.beta * r_k
      
        R = np.stack(self.res_hist, axis=1)
        F = np.stack(self.field_hist, axis=1)

        try:
            c, *_ = np.linalg.lstsq(R, r_k, rcond=1e-8)
        except np.linalg.LinAlgError:
            # if system is singular, use simple mixing
            return w + self.beta * r_k
        
        dw = F @ c

        # Clip update size
        dw_norm = np.linalg.norm(dw)
        if dw_norm > 5 * res_norm:
            dw *= (5 * res_norm / dw_norm)

        return w + self.beta * dw
    
