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
    
    def gaussian_graft(self, width=0.1):
        """
        Returns a smooth initial condition centered at the grafting point.
        width: standard deviation of the Gaussian (in same units as z)
        """
        norm = 1.0 / (width * xp.sqrt(2.0 * xp.pi))
        gaussian = norm * xp.exp(-0.5 * ((self.z - self.grafting_z) / width) ** 2)

        # Normalize so total area = 1 (consistent with delta function)
        gaussian /= xp.sum(gaussian) * self.dz
        return gaussian