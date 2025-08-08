import numpy as np
import matplotlib.pyplot as plt

# System definition
Lz = 50
N = 50
b = 1.0

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

# Build diagonal matrix
from scipy.sparse import diags
from scipy.sparse.linalg import splu
main = -2.0 * np.ones(Nz)
off = 1.0 * np.ones(Nz - 1)
L = diags([off, main, off], offsets=[-1,0,1], shape=(Nz, Nz)).tolil()

# Implement Neuman bc
L[0,0], L[0,1] = -2.0, 2.0
L[-1,-2], L[-1,-1] = 2.0, -2.0

L = L.tocsr()
L /= dz**2
DL = D * L

# =========================================
# Simple grafted chain without interactions
# =========================================

# Provide potential
w = np.zeros_like(z)
W = diags(w,0)

# Setup Crank-Nicolson Matrices
A = DL - W
I = diags([np.ones(Nz)],[0])
M_left = (I - 0.5 * ds * A).tocsc()
M_right = (I + 0.5 * ds * A).tocsc()

# LU factorization to speed up solving
lu = splu(M_left)

# Propagate forward in s
q = q_forward_init.copy()
q_forward = [q.copy()]
for n in range(Ns):
    rhs = M_right @ q
    q = lu.solve(rhs)
    q_forward.append(q.copy())
q_forward = np.array(q_forward)

# Propagate forward in s
q = q_backward_init.copy()
q_backward = [q.copy()]
for n in range(Ns):
    rhs = M_right @ q
    q = lu.solve(rhs)
    q_backward.append(q.copy())
q_backward = np.array(q_backward)

# The partition function
Q = np.trapz(q_forward[-1,:],z)
print("Q = ", Q)

# The polymer density
rho_p = np.zeros_like(z)
for n in range(Ns + 1):
    q_s = q_forward[n]
    qd_s = q_backward[Ns - n]
    rho_p += q_s * qd_s

rho_p *= ds / Q
rho_p

# -----------------
# Plot propagators
# -----------------
# fig, ax = plt.subplots()
# ax.plot(z, q_forward[0,:], ":", label='q(z,0)')
# ax.plot(z, q_forward[1*int(Ns/4),:], ":", label=f'q(z,{1*int(Ns/4)})')
# ax.plot(z, q_forward[2*int(Ns/4),:], ":", label=f'q(z,{2*int(Ns/4)})')
# ax.plot(z, q_forward[3*int(Ns/4),:], ":", label=f'q(z,{3*int(Ns/4)})')
# ax.plot(z, q_forward[4*int(Ns/4),:], ":", label=f'q(z,{4*int(Ns/4)})')

# ax.plot(z, q_backward[0,:], "--", label='q(z,0)')
# ax.plot(z, q_backward[1*int(Ns/4),:], "--", label=f'q(z,{1*int(Ns/4)})')
# ax.plot(z, q_backward[2*int(Ns/4),:], "--", label=f'q(z,{2*int(Ns/4)})')
# ax.plot(z, q_backward[3*int(Ns/4),:], "--", label=f'q(z,{3*int(Ns/4)})')
# ax.plot(z, q_backward[4*int(Ns/4),:], "--", label=f'q(z,{4*int(Ns/4)})')
# ax.set_ylabel('q')
# ax.set_xlabel('z')
# ax.legend()
# plt.show()

# -----------------
# Plot density
# -----------------
print("H = ", np.trapz(rho_p*z,z)/np.trapz(rho_p,z))
fig, ax = plt.subplots()
ax.plot(z, rho_p, "-")
ax.set_ylabel('rho')
ax.set_xlabel('z')
ax.legend()
plt.show()


# ==============================================
# Chain with excluded volume + incompressibility
# ==============================================
max_iter = int(5e3)
tol = 1e-6
mix = 0.02
mix_min = 0.01
mix_max = 1.0
shrink = 0.2
grow = 1.05
prev_dw = None
alpha = 0.2

# Brush parameters
sigma = 0.1
chi = 0.5
rho_p *= sigma
rho_s = 1 - rho_p

# Provide potential
w_p = chi * rho_p  # polymer field
w_s = chi * rho_s  # solvent field
lambda_field = np.zeros_like(z)
mixes = []
res_norms = []
dws = []

from src.dft.solvers import Anderson
mixer = Anderson(m=5,beta=1.0,adaptive_beta=True)

for iteration in range(max_iter):
    W = diags(w_p,0)

    # Setup Crank-Nicolson Matrices
    A = DL - W
    I = diags([np.ones(Nz)],[0])
    M_left = (I - 0.5 * ds * A).tocsc()
    M_right = (I + 0.5 * ds * A).tocsc()

    # LU factorization to speed up solving
    lu = splu(M_left)

    # Propagate forward in s
    q = q_forward_init.copy()
    q_forward = [q.copy()]
    for n in range(Ns):
        rhs = M_right @ q
        q = lu.solve(rhs)
        q_forward.append(q.copy())
    q_forward = np.array(q_forward)

    # Propagate forward in s
    q = q_backward_init.copy()
    q_backward = [q.copy()]
    for n in range(Ns):
        rhs = M_right @ q
        q = lu.solve(rhs)
        q_backward.append(q.copy())
    q_backward = np.array(q_backward)

    Q_p = np.trapz(q_forward[-1,:],z)

    # The polymer density
    rho_p = np.zeros_like(z)
    for n in range(Ns + 1):
        q_s = q_forward[n]
        qd_s = q_backward[Ns - n]
        rho_p += q_s * qd_s
    rho_p *= ds / Q_p
    rho_p *= sigma
    
    # Clip rho values to contrained values
    rho_p = np.where(rho_p >= 1.0, 1.0 - 1e-16, rho_p)
    
    if iteration == 0:
        rho_s = 1 - rho_p
    else:
        # Solvent density and field
        rho_s = np.exp(-w_s)                       

    # Incompressibility residual
    residual = rho_p + rho_s - 1.0
    res_norm = np.linalg.norm(residual) / Nz

    alpha = min(0.01, 0.01 / (res_norm + 1e-10))
    lambda_field += alpha * residual

    rho_p = np.clip(rho_p, 1e-12, 1 - 1e-12)
    rho_s = np.clip(rho_s, 1e-12, 1 - 1e-12)

    # Calculate new fields
    w_p_new = chi * rho_s + lambda_field # np.log(rho_p) + chi * rho_s + lambda_field
    w_s_new = chi * rho_p + lambda_field # np.log(rho_s) + chi * rho_p + lambda_field

    # Check convergence
    dw = np.linalg.norm(w_p_new - w_p) / np.linalg.norm(w_p_new)
    
    print(f"iter {iteration}: dw = {dw:.2e} res = {res_norm:.2e} Q = {Q_p:.4e}")
    if dw < tol and res_norm < 1e-4:
        break

    # Adaptive mixing
    if prev_dw is not None:
        if dw > prev_dw:
            mix = max(mix * shrink, mix_min)
        else:
            mix = min(mix * grow, mix_max)

    # Update fields
    w_p = mixer.mix(w_p, w_p_new)
    w_s = mixer.mix(w_s, w_s_new)
    # w_p = (1 - mix) * w_p + mix * w_p_new 
    # w_s = (1 - mix) * w_s + mix * w_s_new

    prev_dw = dw
    # mixes.append(mix)
    res_norms.append(res_norm)
    dws.append(dw)

q0 = q_forward_init  # initial forward propagator
print("int q0(z) dz =", np.trapz(q_forward_init, z))
total_monomers = np.trapz(rho_p, z)
print("int rho(z) dz =", total_monomers)
print("Expected = sigma * N =", sigma * N)
print("Height = ", np.trapz(z*rho_p,z)/np.trapz(rho_p,z))
print("Q = ", Q_p)
print("This chain in solution would have\nRg = ", np.sqrt((N*(b)**2)/6))

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
ax[2].plot(dws, "-",label='dw')
ax[2].set_yscale('log')
# ax.set_ylim([0,0.1])
ax[2].legend()
plt.show()

