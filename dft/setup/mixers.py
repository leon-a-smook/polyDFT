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

    def mix(self, w, w_new):
        r = w_new - w               # newest residuals

        if len(self.res_hist) >= self.m:
            self.res_hist.pop(0)
            self.field_hist.pop(0)

        self.res_hist.append(r.copy())
        self.field_hist.append(w_new - w)

        if len(self.res_hist) < 2:
            # No history accumulated yet
            return w + self.beta * r
        
        R = np.stack(self.res_hist, axis=1)
        F = np.stack(self.field_hist, axis=1)

        try:
            c, *_ = np.linalg.lstsq(R, r, rcond=None)
        except np.linalg.LinAlgError:
            # if system is singular, use simple mixing
            return w + self.beta * r
        
        dw = F @ c
        return w + self.beta * dw