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

        # # Clip update size
        # dw_norm = np.linalg.norm(dw)
        # if dw_norm > 5 * res_norm:
        #     dw *= (5 * res_norm / dw_norm)

        return w + self.beta * dw
    
class NewtonKrylovSCFT:
    def __init__(self, Nz, z, DL, ds, Ns, qf_init, qb_init, chi,
                 newton_tol=1e-6, max_newton=15, gmres_rtol=1e-3,
                 ls_c=1e-4, ls_min=1e-3, restart=50):
        self.Nz, self.z, self.DL, self.ds, self.Ns = Nz, z, DL, ds, Ns
        self.qf_init, self.qb_init = qf_init, qb_init
        self.chi = chi
        self.newton_tol = newton_tol
        self.max_newton = max_newton
        self.gmres_rtol = gmres_rtol
        self.ls_c, self.ls_min = ls_c, ls_min
        self.restart = restart

    def _split(self, x):
        Nz = self.Nz
        return x[:Nz], x[Nz:2*Nz], x[2*Nz:]

    def _pack(self, wp, ws, lam):
        return np.concatenate([wp, ws, lam])

    def _rho_p_from_wp(self, wp):
        M_right, lu = get_Crank_Nicolson_mat(wp, self.DL, self.ds)
        qf = propagate_q(self.qf_init, M_right, lu, self.Ns)
        qb = propagate_q(self.qb_init, M_right, lu, self.Ns)
        rho_p = get_rho_p(qf, qb, self.z, self.ds, self.Ns)
        return np.clip(rho_p, 1e-12, 1-1e-12)

    def F(self, x):
        wp, ws, lam = self._split(x)
        rho_p = self._rho_p_from_wp(wp)
        rho_s = np.clip(np.exp(-ws), 1e-12, 1-1e-12)
        r1 = wp - self.chi * rho_s - lam
        r2 = ws - self.chi * rho_p - lam
        r3 = rho_p + rho_s - 1.0
        return np.concatenate([r1, r2, r3])

    def _Jv(self, x, Fx, v):
        nv = np.linalg.norm(v)
        if nv == 0: return v.copy()
        # Knoll–Keyes style epsilon
        eps = np.sqrt(np.finfo(float).eps) * (1.0 + np.linalg.norm(x)) / nv
        return (self.F(x + eps * v) - Fx) / eps

    def solve(self, wp0, ws0, lam0, verbose=True):
        x = self._pack(wp0, ws0, lam0)
        Nz = self.Nz

        for k in range(self.max_newton):
            Fx = self.F(x)
            res = np.linalg.norm(Fx) / Nz
            if verbose:
                print(f"[Newton {k}] ||F||/Nz = {res:.3e}")
            if res < self.newton_tol:
                break

            A = LinearOperator(
                dtype=float, shape=(3*Nz, 3*Nz),
                matvec=lambda v: self._Jv(x, Fx, v)
            )
            dx, info = gmres(A, -Fx, rtol=self.gmres_rtol, atol=0.0, restart=self.restart)
            if info not in (0, None):
                # GMRES failed to hit rtol; still try a damped step
                if verbose: print(f"  GMRES info={info}; applying damped step")

            # Backtracking line search on ||F||
            tau = 1.0
            base = np.linalg.norm(Fx)
            while True:
                xt = x + tau * dx
                Ft = self.F(xt)
                if np.linalg.norm(Ft) <= (1 - self.ls_c * tau) * base or tau < self.ls_min:
                    x = xt
                    break
                tau *= 0.5

        return self._split(x), {"iters": k+1, "final_res": np.linalg.norm(self.F(x))/Nz}
    
# src/dft/solvers.py
import time
import numpy as np
from dataclasses import dataclass, field
from scipy.sparse.linalg import LinearOperator, gmres
from .core import get_Crank_Nicolson_mat, propagate_q, get_rho_p

@dataclass
class SCFTStats:
    # per-Newton-step logs
    res_norms: list = field(default_factory=list)
    gmres_iters: list = field(default_factory=list)
    gmres_restarts: list = field(default_factory=list)
    ls_halvings: list = field(default_factory=list)
    step_alpha: list = field(default_factory=list)
    r1_norm: list = field(default_factory=list)
    r2_norm: list = field(default_factory=list)
    r3_norm: list = field(default_factory=list)
    clipped_p_frac: list = field(default_factory=list)
    clipped_s_frac: list = field(default_factory=list)
    reduction_ratio: list = field(default_factory=list)

    # totals and timings
    F_calls: int = 0
    Jv_calls: int = 0
    LU_count: int = 0
    t_F: float = 0.0
    t_lu: float = 0.0
    t_prop: float = 0.0
    t_gmres: float = 0.0
    t_linesearch: float = 0.0
    t_total: float = 0.0

class NewtonKrylovSCFT_Timed:
    def __init__(self, Nz, z, DL, ds, Ns, qf_init, qb_init, chi,
                 newton_tol=1e-6, max_newton=15, gmres_rtol=1e-3,
                 ls_c=1e-4, ls_min=1e-3, restart=50, stats=None, verbose=True):
        self.Nz, self.z, self.DL, self.ds, self.Ns = Nz, z, DL, ds, Ns
        self.qf_init, self.qb_init = qf_init, qb_init
        self.chi = chi
        self.newton_tol = newton_tol
        self.max_newton = max_newton
        self.gmres_rtol = gmres_rtol
        self.ls_c, self.ls_min = ls_c, ls_min
        self.restart = restart
        self.stats = stats if stats is not None else SCFTStats()
        self.verbose = verbose

    def _split(self, x):
        Nz = self.Nz
        return x[:Nz], x[Nz:2*Nz], x[2*Nz:]

    def _pack(self, wp, ws, lam):
        return np.concatenate([wp, ws, lam])

    def _rho_p_from_wp(self, wp, timers):
        t0 = time.perf_counter()
        t_lu0 = time.perf_counter()
        M_right, lu = get_Crank_Nicolson_mat(wp, self.DL, self.ds)
        self.stats.LU_count += 1
        t_lu1 = time.perf_counter()
        qf = propagate_q(self.qf_init, M_right, lu, self.Ns)
        qb = propagate_q(self.qb_init, M_right, lu, self.Ns)
        t1 = time.perf_counter()
        self.stats.t_lu += (t_lu1 - t_lu0)
        self.stats.t_prop += (t1 - t_lu1)
        timers['t_lu'] += (t_lu1 - t_lu0)
        timers['t_prop'] += (t1 - t_lu1)
        rho_p = get_rho_p(qf, qb, self.z, self.ds, self.Ns)
        return np.clip(rho_p, 1e-12, 1-1e-12)

    def F(self, x):
        self.stats.F_calls += 1
        t0 = time.perf_counter()
        timers = {'t_lu':0.0, 't_prop':0.0}

        wp, ws, lam = self._split(x)
        rho_p = self._rho_p_from_wp(wp, timers)
        rho_s = np.clip(np.exp(-ws), 1e-12, 1-1e-12)

        r1 = wp - self.chi * rho_s - lam
        r2 = ws - self.chi * rho_p - lam
        r3 = rho_p + rho_s - 1.0

        self.stats.t_lu += 0.0  # already added above
        self.stats.t_prop += 0.0
        self.stats.t_F += time.perf_counter() - t0
        # keep component norms for the last call by solve()
        self._last_component_norms = (np.linalg.norm(r1)/self.Nz,
                                      np.linalg.norm(r2)/self.Nz,
                                      np.linalg.norm(r3)/self.Nz)
        # clipping fractions (diagnostic)
        self._last_clip_frac = ((rho_p<=1e-12).mean() + (rho_p>=1-1e-12).mean(),
                                (rho_s<=1e-12).mean() + (rho_s>=1-1e-12).mean())
        return np.concatenate([r1, r2, r3])

    def _Jv(self, x, Fx, v):
        self.stats.Jv_calls += 1
        nv = np.linalg.norm(v)
        if nv == 0: return v.copy()
        eps = np.sqrt(np.finfo(float).eps) * (1.0 + np.linalg.norm(x)) / nv
        return (self.F(x + eps * v) - Fx) / eps

    def solve(self, wp0, ws0, lam0):
        x = self._pack(wp0, ws0, lam0)
        Nz = self.Nz
        t_all0 = time.perf_counter()

        for k in range(self.max_newton):
            Fk = self.F(x)
            res = np.linalg.norm(Fk) / Nz
            self.stats.res_norms.append(res)
            if self.verbose:
                print(f"[Newton {k}] ||F||/Nz = {res:.3e}")

            if res < self.newton_tol:
                break

            # GMRES with iteration counter
            iters = 0
            def cb(_):
                nonlocal iters
                iters += 1

            A = LinearOperator(
                dtype=float, shape=(3*Nz, 3*Nz),
                matvec=lambda v: self._Jv(x, Fk, v)
            )
            t_gm0 = time.perf_counter()
            dx, info = gmres(A, -Fk, rtol=self.gmres_rtol, atol=0.0,
                             restart=self.restart, callback=cb)
            t_gm1 = time.perf_counter()
            self.stats.t_gmres += (t_gm1 - t_gm0)
            self.stats.gmres_iters.append(iters)
            self.stats.gmres_restarts.append(0 if info in (0, None) else 1)

            # Backtracking
            base = np.linalg.norm(Fk)
            tau = 1.0
            halvings = 0
            t_ls0 = time.perf_counter()
            while True:
                Ft = self.F(x + tau * dx)
                if np.linalg.norm(Ft) <= (1 - self.ls_c * tau) * base or tau < self.ls_min:
                    break
                tau *= 0.5
                halvings += 1
            self.stats.t_linesearch += (time.perf_counter() - t_ls0)
            self.stats.ls_halvings.append(halvings)
            self.stats.step_alpha.append(tau)
            self.stats.reduction_ratio.append(np.linalg.norm(Ft)/ (base + 1e-300))
            self.stats.r1_norm.append(self._last_component_norms[0])
            self.stats.r2_norm.append(self._last_component_norms[1])
            self.stats.r3_norm.append(self._last_component_norms[2])
            self.stats.clipped_p_frac.append(self._last_clip_frac[0])
            self.stats.clipped_s_frac.append(self._last_clip_frac[1])

            x = x + tau * dx

        self.stats.t_total = time.perf_counter() - t_all0
        return self._split(x), {"iters": k+1, "final_res": self.stats.res_norms[-1]}
