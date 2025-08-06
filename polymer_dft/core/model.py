# core/model.py
from core.arrays import xp

def compute_monomer_density(q, q_dag, Q, config, grid):
    """
    Compute monomer density profile from forward and backward propagators.
    """
    N = config.N
    dz = grid.dz
    Ns = int(N / config.ds)

    rho = xp.zeros_like(q[:, 0])
    for s in range(Ns):
        rho += q[:, s] * q_dag[:, Ns - 1 - s]
    rho *= 1 / (Q * Ns)

    return rho

def compute_brush_height(rho, grid):
    z = grid.z
    dz = grid.dz
    total = xp.sum(rho) * dz
    moment = xp.sum(rho * z) * dz
    return moment / total


def wall_potential(z, amplitude=10.0, decay_length=0.2):
    """
    Returns a soft wall potential that decays exponentially from z = 0.
    Units: same as z.
    """
    return amplitude * xp.exp(-z / decay_length)
