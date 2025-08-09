import numpy as np
import matplotlib.pyplot as plt

# System definition
Lz = 50
N = 50
b = 1.0
chi = 0.0
max_iter = 300
tol = 1e-6
mix = 0.2
sigma = 0.1
alpha = 0.1
# Choose discretization
dz = b / 8.0 
Nz = int(Lz / dz) + 1
z = np.linspace(0, Lz, num=Nz)

# Chain definition
ds = 0.04 
Ns = int(N / ds)
s_vals = np.linspace(0, N, Ns+1)

# System parameters
D = (b**2 / 6)     
print("This chain in solution would have\nRg = ", b*np.sqrt(N/6))
print("The following value should be < 0.5: ", D*ds/(dz**2))
assert D*ds/dz/dz <= 0.5, "Adjust discretization"

# Set up forward propagator
q_forward_init = np.zeros_like(z)
q_forward_init[1] = 1 / dz

# Set up backward propagator
q_backward_init = np.zeros_like(z)
q_backward_init[:] = 1

from src.dft.core import get_Lagrange_mat, compute_rho_p
from src.dft.solvers import Anderson

w_p = np.zeros_like(z)
w_s = np.zeros_like(z)

DL = get_Lagrange_mat(Nz, dz, D)

lambda_field = np.zeros_like(z)
res_norms = []
dws = []

anderson = Anderson(m=20,beta=1.0,adaptive_beta=True)
for iteration in range(max_iter):
    rho_p = compute_rho_p(w_p, z, DL, ds, Ns, q_forward_init, q_backward_init, sigma)
    rho_s = np.exp(-w_s) 

    # Incompressibility residual
    residual = rho_p + rho_s - 1.0
    res_norm = np.linalg.norm(residual) / Nz

    # alpha = min(0.1, 0.1 / (res_norm + 1e-12))
    lambda_field += alpha * residual

    # Calculate new fields
    w_p_new = chi * rho_s + lambda_field 
    w_s_new = chi * rho_p + lambda_field 

    # Check convergence
    dw = np.linalg.norm(w_p_new - w_p) / np.linalg.norm(w_p_new)
    
    print(f"iter {iteration}: dw = {dw:.2e} res = {res_norm:.2e}")
    if dw < tol and res_norm < 1e-4:
        print(f"Converged in {iteration} iterations.")
        break
    
    
    # Update fields
    w_all = np.concatenate((w_p, w_s))
    w_all_new = np.concatenate((w_p_new, w_s_new))
    w_mixed = anderson.mix(w_all, w_all_new)
    w_p = w_mixed[:Nz]
    w_s = w_mixed[Nz:]
    # w_p = anderson.mix(w_p, w_p_new)
    # w_s = anderson.mix(w_s, w_s_new)

    # Stats tracking
    res_norms.append(res_norm)
    dws.append(dw)

fig, ax = plt.subplots(ncols=2, figsize=(7,3))
ax[0].plot(z, rho_p, "-", label='Polymer')
ax[0].plot(z, rho_s, "-", label="Solvent")
ax[0].set_ylabel('$\\rho$')
ax[0].set_xlabel('z')
# ax.set_ylim([0,0.1])
ax[0].legend()

ax[1].plot(res_norms, "-", label="res_norm")
ax[1].plot(dws, "-",label='dw')
ax[1].set_yscale('log')
# ax.set_ylim([0,0.1])
ax[1].legend()
plt.show()
