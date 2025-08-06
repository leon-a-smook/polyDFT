# tests/test_crank_nicolson.py

import pytest
from core.grid import Grid1D
from numerics.finite_diff import crank_nicolson_step
from core.arrays import xp, to_numpy

class DummyConfig:
    def __init__(self):
        self.a2 = 1.0
        self.ds = 0.01

@pytest.fixture
def setup_grid():
    grid = Grid1D(Lz=10.0, Nz=256, grafting_z=1.0)
    config = DummyConfig()
    potential = xp.zeros_like(grid.z)
    return grid, config, potential

def test_propagator_normalization(setup_grid):
    grid, config, potential = setup_grid
    q = grid.delta_graft()

    for _ in range(50):
        q = crank_nicolson_step(q, potential, grid, config)
        total = float(xp.sum(q) * grid.dz)
        assert abs(total - 1.0) < 1e-3

@pytest.mark.parametrize("do_plot", [False])  # Toggle to True manually if desired
def test_visual_propagation(setup_grid, do_plot):
    if not do_plot:
        pytest.skip("Skipping plot test (toggle --plot to enable)")

    import matplotlib.pyplot as plt
    grid, config, potential = setup_grid
    q = grid.gaussian_graft(width=0.2)

    q_all = [q]
    for _ in range(50):
        q = crank_nicolson_step(q, potential, grid, config)
        q_all.append(q)

    z = to_numpy(grid.z)
    for i, qi in enumerate(q_all[::10]):
        plt.plot(z, to_numpy(qi), label=f"s={i*10}")
    plt.xlabel("z")
    plt.ylabel("q(z, s)")
    plt.title("Crank–Nicolson Propagation (Visual)")
    plt.legend()
    plt.tight_layout()
    plt.show()