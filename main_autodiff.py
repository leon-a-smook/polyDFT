# main_autodiff.py
# A main loop using automatic differentiation with a Newton solver
# to find the self-consistent solution.

# ----------------
# Imports
# ----------------
import jax
import jax.numpy as jnp
from jax.scipy.linalg import solve
import matplotlib.pyplot as plt

# ===================
# Setup
# ===================
# User input
# ----------------
# Physical parameters
N = 50                              # Chain length (Test value: 32)
b = 1.0                             # Kuhn length
ds = 0.04                           # Chain discretization
Lz = 50                             # Domain size (Test value: 32)
dz = b / 8.0                        # Domain discretization
sigma = 0.1                         # Grafting density (z^-1)
chi = 0.0                           # Flory-Huggins parameter

# Solver settings
max_iter = 20
tol = 1e-6
mix = 0.02
mix_min = 0.01
mix_max = 1.0
shrink = 0.2
grow = 1.05
prev_dw = None
alpha = 0.2
adaptive_alpha = True

# Computed initialization
# --------------------
# Computed parameters
D = (b**2 / 6)                      # Chain 'diffusion' in propagator

# Grid
Ns = int(N / ds)                    # Number of chain segments
s_vals = jnp.linspace(0, N, Ns+1)   # Chain 'grid'
Nz = int(Lz / dz) + 1               # Number of domain grid points
z = jnp.linspace(0, Lz, num=Nz)     # Domain grid
grid_alpha = D*ds/(dz**2)           # Crank-Nicolson stability parameter

# Discretization checks
print(f"Discretization: grid_alpha = {grid_alpha:.3f} (< 0.5 required)")
assert grid_alpha <= 0.5, "Stability parameter not < 0.5. Adjust discretization by increasing ds or decreasing dz."
print(f"This chain in solution would have\nRg =    {b*jnp.sqrt(N/6):.3f}")

# Initialization of propagators
q_forward_init = jnp.zeros_like(z).at[1].set(1/dz)   # Forward propagator
q_backward_init = jnp.ones_like(z)                   # Backward propagator

# =================================
# Solve modified Edwards-Diffusion
# =================================

# Construct Lagrangian matrix
main = -2.0 * jnp.ones(Nz)                          # Diagonal element
off = 1.0 * jnp.ones(Nz - 1)                        # Off-diagonal element

L = jnp.zeros((Nz, Nz))                             # Allocate space for Lagrangian
d_ix = jnp.diag_indices(Nz)
# Set off-diagonals
L = L.at[d_ix[0][1:],d_ix[1][:-1]].set(off).at[d_ix[0][:-1],d_ix[1][1:]].set(off)    
# Set main diagonal     
L = L.at[d_ix].set(main)                            

# Implement Neumann boundary conditions
L = L.at[0, 0].set(-2.0).at[0, 1].set(2.0)  # BC at the start
L = L.at[-1, -2].set(2.0).at[-1, -1].set(-2.0)  # BC at the end
L /= dz**2  # Normalize by dz^2
DL = D * L

# Implement potential
w = jnp.zeros_like(z)
W = jnp.diag(w)

# Setup Crank-Nicolson matrices
A = DL - W
I = jnp.eye(Nz)
M_left = I - 0.5 * ds * A
M_right = I + 0.5 * ds * A

# Forward propagator
q = q_forward_init.copy()
q_forward = [q.copy()]
for n in range(Ns):
    rhs = M_right @ q
    q = solve(M_left, rhs)
    q_forward.append(q.copy())
q_forward = jnp.array(q_forward)

# Backward propagator
q = q_backward_init.copy()
q_backward = [q.copy()]
for n in range(Ns):
    rhs = M_right @ q
    q = solve(M_left, rhs)
    q_backward.append(q.copy())
q_backward= jnp.array(q_backward)

# The partition function
Q = jnp.trapezoid(q_forward[-1,:],z)
print(f"Q =     {Q:.3f}")

# The polymer density
rho_p = jnp.zeros_like(z)
for n in range(Ns + 1):
    q_s = q_forward[n,:]
    qd_s = q_backward[Ns - n,:]
    rho_p += q_s * qd_s
rho_p *= ds / Q
print(f"H0 =    {jnp.trapezoid(rho_p*z,z)/jnp.trapezoid(rho_p,z):.3f}")

# =============================================
# Self-consistent loop using Newton's method
# =============================================

# Initial fields
w_p = jnp.zeros_like(z)             # Polymer
w_s = jnp.zeros_like(z)             # Solvent
lambda_field = jnp.zeros_like(z)    # Lagrange multipliers
dw_ps, dw_ss, res_norms = [], [], []

# Main solver loop
for iteration in range(max_iter):
    # Crank-Nicolson matrices
    W = jnp.diag(w_p)
    A = DL - W
    A = DL - W
    I = jnp.eye(Nz)
    M_left = I - 0.5 * ds * A
    M_right = I + 0.5 * ds * A

    # Forward propagator
    q = q_forward_init.copy()
    q_forward = [q.copy()]
    for n in range(Ns):
        rhs = M_right @ q
        q = solve(M_left, rhs)
        q_forward.append(q.copy())
    q_forward = jnp.array(q_forward)

    # Backward propatagor
    q = q_backward_init.copy()
    q_backward = [q.copy()]
    for n in range(Ns):
        rhs = M_right @ q
        q = solve(M_left, rhs)
        q_backward.append(q.copy())
    q_backward= jnp.array(q_backward)

    # Polymer density from propagators
    Q_p = jnp.trapezoid(q_forward[-1], z)
    rho_p = jnp.zeros_like(z)
    for n in range(Ns + 1):
        q_s = q_forward[n]
        qd_s = q_backward[Ns - n]
        rho_p += q_s * qd_s
    rho_p *= ds / Q_p
    rho_p *= sigma

    # Clip the density to constrained values
    rho_p = jnp.clip(rho_p, 0, 1)

    # Solvent density and incompressibility
    rho_s = jnp.exp(-w_s)  # Solvent density and field

    plt.title("Densities")
    plt.plot(z, rho_p, label='$\\rho_{p}$$')
    plt.plot(z, rho_s, label='$\\rho_{s}$$')
    plt.show()

    # Compute residuals and update lambda field
    residual = rho_p + rho_s - 1.0
    if jnp.any(jnp.isnan(residual)) or jnp.any(jnp.isinf(residual)):
            print(f"Warning: Residual has NaN or Inf values at iteration {iteration}")
            break

    # Update Lagrange field
    res_norm = jnp.linalg.norm(residual) / Nz
    if adaptive_alpha:
        alpha = jnp.minimum(0.1, 0.1 / (jnp.linalg.norm(residual) + 1e-10))
    lambda_field += alpha * residual

    # Update the fields
    w_p_new = chi * rho_s + lambda_field
    w_s_new = chi * rho_p + lambda_field

    delta_w_p = w_p - w_p_new
    delta_w_s = w_s - w_s_new

    plt.title("Delta w_s")
    plt.plot(z, delta_w_p, label='$w_{p}$$')
    plt.plot(z, delta_w_s, label='$w_{s}$$')
    plt.legend()
    plt.show()

    dw_p = jnp.linalg.norm(delta_w_p) / jnp.linalg.norm(w_p_new)
    dw_s = jnp.linalg.norm(delta_w_s) / jnp.linalg.norm(w_s_new)

    if dw_p < tol and res_norm < 1e-4 and dw_s < tol:
        print(f"Converged in {iteration} iterations")
        break

    # Compute Jacobian 
    jac_res_p = jax.jacobian(lambda w: delta_w_p)
    jac_val_p = jac_res_p(w_p)
    print(jac_val_p)
    delta_w_p = solve(jac_val_p, delta_w_p)
    print(delta_w_p)
    w_p += delta_w_p
    jac_res_s = jax.jacobian(lambda w: delta_w_s)
    jac_val_s = jac_res_s(w_s)
    delta_w_s = solve(jac_val_s, -delta_w_s)
    w_s += delta_w_s

    plt.title("w_s")
    plt.plot(z, w_p, label='$w_{p}$$')
    plt.plot(z, w_s, label='$w_{s}$$')
    plt.legend()
    plt.show()

    print(f"Iteration {iteration}: dw_p = {dw_p:.2e}, dw_s = {dw_s:.2e}, res_norm = {res_norm:.2e}")

    dw_ps.append(dw_p)  # Store the current dw for the next iteration
    dw_ss.append(dw_s)
    res_norms.append(res_norm)

q0 = q_forward_init  # initial forward propagator
print("int q0(z) dz =", jnp.trapezoid(q_forward_init, z))
total_monomers = jnp.trapezoid(rho_p, z)
print("int rho(z) dz =", total_monomers)
print("Expected = sigma * N =", sigma * N)
print("Height = ", jnp.trapezoid(z*rho_p,z)/jnp.trapezoid(rho_p,z))
print("Q = ", Q_p)
print("This chain in solution would have\nRg = ", jnp.sqrt((N*(b)**2)/6))

# -----------------
# Plot density
# -----------------
fig, ax = plt.subplots(ncols=3, figsize=(10,3))
ax[0].plot(z, rho_p, "-", label='Polymer')
ax[0].plot(z, rho_s, "-", label="Solvent")
ax[0].set_ylabel('$\\rho$')
ax[0].set_xlabel('z')
# ax.set_ylim([0,0.1])
ax[0].legend()

ax[1].plot(z, residual, "-", label='Residual')
ax[1].plot(z, w_p_new - w_p, "-", label="dw_pol")
ax[1].set_ylabel('Value')
ax[1].set_xlabel('z')
# ax.set_ylim([0,0.1])
ax[1].legend()

# ax[2].plot(mixes, "-", label='mix')
ax[2].plot(res_norms, "-", label="res_norm")
ax[2].plot(dw_ss, "-",label='dw_s')
ax[2].plot(dw_ps, "-",label='dw_p')
ax[2].set_yscale('log')
# ax.set_ylim([0,0.1])
ax[2].legend()
plt.show()

