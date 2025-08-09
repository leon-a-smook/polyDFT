# dft/core.py
# Core functionality in the dft loops
import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import splu
from numba import jit


def compute_rho_p(w_p, z, DL, ds, Ns, q_forward_init, q_backward_init, sigma=1.0, eps_rho=1e-16):
    M_right , lu = get_Crank_Nicolson_mat(w_p, DL, ds)
    q_forward = propagate_q(q_forward_init, M_right, lu, Ns)
    q_backward = propagate_q(q_backward_init, M_right, lu, Ns)
    rho_p = get_rho_p(q_forward,q_backward,z,ds,Ns, sigma=sigma)
    rho_p = np.where(rho_p >= 1.0, 1.0 - eps_rho, rho_p)
    return rho_p

def get_Crank_Nicolson_mat(w_p, DL, ds):
    Nz = len(w_p)
    W = diags(w_p,0)
    # Setup Crank-Nicolson Matrices
    A = DL - W
    I = diags([np.ones(Nz)],[0])
    M_left = (I - 0.5 * ds * A).tocsc()
    M_right = (I + 0.5 * ds * A).tocsc()

    # LU factorization to speed up solving
    lu = splu(M_left)
    return M_right, lu

def get_Lagrange_mat(Nz, dz, D):
    main = -2.0 * np.ones(Nz)
    off = 1.0 * np.ones(Nz - 1)
    L = diags([off, main, off], offsets=[-1,0,1], shape=(Nz, Nz)).tolil()
    # Implement Neuman bc
    L[0,0], L[0,1] = -2.0, 2.0
    L[-1,-2], L[-1,-1] = 2.0, -2.0
    L = L.tocsr()
    L /= dz**2
    DL = D * L
    return DL

def propagate_q(q_init, M_right, lu, Ns):
    q = q_init.copy()
    q_mat = [q.copy()]
    for n in range(Ns):
        rhs = M_right @ q
        q = lu.solve(rhs)
        q_mat.append(q.copy())
    return(np.array(q_mat))

def get_rho_p(q_forward, q_backward, z, ds, Ns, sigma=1.0):
    # Partition function
    Q = np.trapz(q_forward[-1,:],z)

    # The polymer density
    rho_p = np.zeros_like(z)
    for n in range(Ns + 1):
        q_s = q_forward[n]
        qd_s = q_backward[Ns - n]
        rho_p += q_s * qd_s

    rho_p *= ds / Q
    rho_p *= sigma
    return rho_p