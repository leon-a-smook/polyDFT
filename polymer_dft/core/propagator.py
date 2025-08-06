# core/propagator.py

from numerics.finite_diff import crank_nicolson_step
from core.arrays import xp

def compute_propagators(w, grid, config):
    """
    Computes:
      - Forward propagator q(z, s) from s=0 to N
      - Backward propagator q_dag(z, s) from s=N to 0
      - Partition function Q = ∫ q(z, N) dz

    Returns:
        q: (Nz, Ns) array
        q_dag: (Nz, Ns) array
        Q: scalar
    """
    Nz = grid.Nz
    Ns = int(config.N / config.ds)
    dz = grid.dz

    # Forward propagator
    q = xp.zeros((Nz, Ns))
    q = q.at[:, 0].set(grid.gaussian_graft(width=0.05))

    for s in range(1, Ns):
        q = q.at[:, s].set(crank_nicolson_step(q[:, s - 1], w, grid, config))

    # Backward propagator
    q_dag = xp.ones((Nz, Ns))  # initialize with 1
    for s in range(1, Ns):
        q_dag = q_dag.at[:, s].set(
            crank_nicolson_step(q_dag[:, s - 1], w, grid, config)
        )

    # Partition function (single-chain)
    Q = xp.sum(q[:, -1]) * dz
    return q, q_dag, Q