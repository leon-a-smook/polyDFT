# main.py
from core.grid import Grid1D
from core.arrays import get_device_info, to_numpy, xp
from core.propagator import compute_propagators
from core.model import compute_brush_height, compute_monomer_density, wall_potential
from numerics.finite_diff import crank_nicolson_step

import matplotlib.pyplot as plt


class DummyConfig:
    def __init__(self):
        self.a2 = 1.0
        self.ds = 0.1
        self.N = 100

def main():
    print("Device info:", get_device_info())

    grid = Grid1D(Lz=30.0, Nz=250, grafting_z=0.1)
    delta = grid.delta_graft()
    gaussian = to_numpy(grid.gaussian_graft(width=0.5))
    config = DummyConfig()

    # # Plot δ(z)
    # z = to_numpy(grid.z)
    # δ = to_numpy(delta)
    # g = to_numpy(gaussian)
    # plt.plot(z, δ, label="δ(z - z₀)")
    # plt.plot(z, g, label="Gaussian (σ = 0.05)")
    # plt.xlabel("z")
    # plt.ylabel("Initial q(z,0)")
    # plt.title("Initial grafting profiles")
    # plt.legend()
    # plt.tight_layout()
    # plt.show()

    z = grid.z
    potential_w = wall_potential(z, amplitude=20, decay_length=0.2)

    # plt.plot(to_numpy(z), to_numpy(potential_w))
    # plt.title("Wall Potential at z=0")
    # plt.xlabel("z")
    # plt.ylabel("w(z)")
    # plt.tight_layout()
    # plt.show()

    q, q_dag, Q = compute_propagators(potential_w, grid, config)
    rho_polymer = compute_monomer_density(q, q_dag, Q, config, grid)
    height = compute_brush_height(rho_polymer, grid)

    z = to_numpy(grid.z)
    plt.plot(z, to_numpy(rho_polymer))
    plt.xlabel("z")
    plt.ylabel("ρ_P(z)")
    plt.title(f"Monomer density, brush height ≈ {float(height):.2f}")
    plt.show()

def test_crank_nicolson():
    grid = Grid1D(Lz=10.0, Nz=1000, grafting_z=0.15)
    config = DummyConfig()

    q_init = grid.gaussian_graft()  # initial condition
    potential = xp.zeros_like(grid.z)

    # Propagate for a few steps
    q = q_init
    q_all = [q]
    for _ in range(50):
        q = crank_nicolson_step(q, potential, grid, config)
        q_all.append(q)

    # Plot evolution
    import matplotlib.pyplot as plt
    for i, q_snap in enumerate(q_all[::10]):
        plt.plot(to_numpy(grid.z), to_numpy(q_snap), label=f"s={i*10}")
    plt.xlabel("z")
    plt.ylabel("q(z, s)")
    plt.title("Crank–Nicolson propagation (test)")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()
    # test_crank_nicolson()
    