# mixers.py
import numpy as np


class Anderson:
    def __init__(self, m=5, beta=0.1):
        self.m = m                  # history depth
        self.beta = beta            # mixing damping
        self.reset()
    
    def reset(self):
        self.res_hist = []          # residuals: r_k = w_new - w
        self.field_hist = []        # field differences: dw(k) = w(k) - w(k-1)
        self.w_prev = None          # stores previous w_k
        self.prev_res_norm = None   # residuals norm previous step

    def mix(self, w, w_new):
        r_k = w_new - w               # newest residuals

        if self.w_prev is not None:
            f_k = w - self.w_prev
            self.field_hist.append(f_k.copy())
            self.res_hist.append(r_k.copy())

            # remove old entries to maintain history size
            if len(self.res_hist) >= self.m:
                self.res_hist.pop(0)
                self.field_hist.pop(0)

        self.w_prev = w.copy()

        res_norm = np.linalg.norm(r_k)
        if self.prev_res_norm is not None and res_norm > self.prev_res_norm * 1.05:
            return w + self.beta * r_k  # fallback to simple mixing
        self.prev_res_norm = res_norm

        if len(self.res_hist) < 1:
            return w + self.beta * r_k
      
        R = np.stack(self.res_hist, axis=1)
        F = np.stack(self.field_hist, axis=1)

        try:
            c, *_ = np.linalg.lstsq(R, r_k, rcond=None)
        except np.linalg.LinAlgError:
            # if system is singular, use simple mixing
            return w + self.beta * r_k
        
        dw = F @ c

        # Clip update size
        dw_norm = np.linalg.norm(dw)
        if dw_norm > 10 * res_norm:
            dw *= (10 * res_norm / dw_norm)
            
        return w + self.beta * dw