# numerics/finite_diff.py
from core.arrays import xp
import jax.numpy.linalg as linalg

def crank_nicolson_step(q_curr, potential_w, grid, config):
    """
    Compute one Crank–Nicolson step in s-direction.
    q_curr: propagator at current s
    potential_w: self-consistent field (1D array)
    grid: Grid1D object
    config: dict or object with a2, ds, etc
    """
    dz = grid.dz
    Nz = grid.Nz
    ds = config.ds
    a2 = config.a2  # typically 1.0

    # Coefficient for Laplacian term
    D = a2 / 6.0
    alpha = D * ds / dz**2

    # Accuracy-check
    assert alpha >= 1, f"alpha = {alpha:.4f}" 

    # Construct tridiagonal matrices (I ± alpha * L)
    main_diag = 1.0 + alpha + 0.5 * ds * potential_w
    off_diag = -0.5 * alpha * xp.ones(Nz - 1)

    # Construct sparse-like banded matrix
    A = tridiag_matrix_with_neumann(main_diag, off_diag)

    # RHS: (I - A) @ q_curr
    main_diag_rhs = 1.0 - alpha - 0.5 * ds * potential_w
    A_rhs = tridiag_matrix_with_neumann(main_diag_rhs, -off_diag)

    rhs = A_rhs @ q_curr
    q_next = linalg.solve(A, rhs)

    # # Log solver output with negative values
    # if xp.any(q_next < 0):
    #     print("Warning: negative propagator values detected.")
    q_next = xp.maximum(q_next, 0.0)
    return q_next

def tridiag_matrix_with_dirichlet(main, off):
    """
    Build tridiagonal matrix: used for Laplacian operator.
    Implement (implicit) dirichlet zero boundary conditions.
    main: main diagonal (1D array)
    off: off diagonals (1D array of length len(main) - 1)
    """
    Nz = main.shape[0]
    mat = xp.zeros((Nz, Nz))
    mat = mat.at[xp.arange(Nz), xp.arange(Nz)].set(main)
    mat = mat.at[xp.arange(Nz - 1), xp.arange(1, Nz)].set(off)
    mat = mat.at[xp.arange(1, Nz), xp.arange(Nz - 1)].set(off)
    return mat

def tridiag_matrix_with_neumann(main, off):
    """
    Build tridiagonal matrix: used for Laplacian operator.
    Implement reflecting Neumann boundaries.
    main: main diagonal (1D array)
    off: off diagonals (1D array of length len(main) - 1)
    """
    Nz = main.shape[0]
    mat = xp.zeros((Nz, Nz))

    # Interior
    mat = mat.at[xp.arange(Nz), xp.arange(Nz)].set(main)
    mat = mat.at[xp.arange(Nz - 1), xp.arange(1, Nz)].set(off)
    mat = mat.at[xp.arange(1, Nz), xp.arange(Nz - 1)].set(off)

    # Left boundary (Neumann): modify first row
    mat = mat.at[0, 0].set(main[0] + off[0])
    mat = mat.at[0, 1].set(-off[0])

    # Right boundary (Neumann): modify last row
    mat = mat.at[-1, -1].set(main[-1] + off[-1])
    mat = mat.at[-1, -2].set(-off[-1])

    return mat