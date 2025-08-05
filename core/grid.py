# core/grid.py
from core.arrays import xp

class Grid1D:
    def __init__(self, Lz=20.0, Nz=256, grafting_z=0.2):
        self.Lz = Lz           # Domain length in z-direction
        self.Nz = Nz           # Number of spatial points
        self.dz = Lz / (Nz - 1)
        self.z = xp.linspace(0.0, Lz, Nz)
        self.grafting_z = grafting_z  # Grafting location

    def delta_graft(self):
        # Discretized delta function: 1 in closest bin to graft point
        idx = xp.argmin(xp.abs(self.z - self.grafting_z))
        delta = xp.zeros_like(self.z)
        delta = delta.at[idx].set(1.0 / self.dz)
        return delta