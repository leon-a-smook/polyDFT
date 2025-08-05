# main.py
from core.grid import Grid1D
from core.arrays import get_device_info, to_numpy
import matplotlib.pyplot as plt

def main():
    print("Device info:", get_device_info())

    grid = Grid1D(Lz=10.0, Nz=512, grafting_z=0.2)
    delta = grid.delta_graft()

    # Plot δ(z)
    z = to_numpy(grid.z)
    δ = to_numpy(delta)
    plt.plot(z, δ, label="δ(z - z₀)")
    plt.xlabel("z")
    plt.ylabel("δ(z)")
    plt.title("Discretized delta function (grafting point)")
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()