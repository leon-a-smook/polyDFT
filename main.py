import numpy as np
import matplotlib.pyplot as plt

# System definition
Lz = 40
N = 64
b = 1.0

# Choose discretization
dz = b / 8.0
Nz = int(Lz / dz) + 1
z = np.linspace(0, Lz, num=Nz)

# Chain definition
ds = 0.01
Ns = int(N / ds)
s_vals = np.linspace(0, N, Ns+1)

# System parameters
D = b**2 / 6
print("This chain in solution would have\nRg = ", b*np.sqrt(N/6))

# Set up forward propagator
q_forward_init = np.zeros_like(z)
q_forward_init[1] = 1/dz

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
q = q_forward_init
q_forward = [q.copy()]
for n in range(Ns):
    rhs = M_right @ q
    q = lu.solve(rhs)
    q_forward.append(q.copy())
q_forward = np.array(q_forward)

# Propagate forward in s
q = q_backward_init
q_backward = [q.copy()]
for n in range(Ns):
    rhs = M_right @ q
    q = lu.solve(rhs)
    q_backward.append(q.copy())
q_backward = np.array(q_backward)

# The partition function
Q = np.trapezoid(q_forward[-1,:],z)
print("Q = ", Q)

# The polymer density
rho = np.zeros_like(z)
for n in range(Ns + 1):
    q_s = q_forward[n]
    qd_s = q_backward[Ns - n]
    rho += q_s * qd_s

rho *= ds / Q

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
# fig, ax = plt.subplots()
# ax.plot(z, rho, "-")
# ax.set_ylabel('rho')
# ax.set_xlabel('z')
# ax.legend()
# plt.show()


# ==============================================
# Chain with excluded volume + incompressibility
# ==============================================
max_iter = int(1e3)
tol = 1e-6
mix = 0.1
mix_min = 0.02
mix_max = 0.3
shrink = 0.5
grow = 1.05
prev_dw = None
alpha = 0.02


# Brush parameters
sigma = 0.1
chi = 0.5
V_s = Lz - sigma*N

# Provide potential
w_p = np.zeros_like(z) + 1e-12 # polymer field
w_s = np.zeros_like(z) + 1e-12 # solvent field
lambda_field = np.zeros_like(z)
# from dft.mixers import Anderson
# mixer = Anderson(m=5, beta=0.02)

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
    q = q_forward_init
    q_forward = [q.copy()]
    for n in range(Ns):
        rhs = M_right @ q
        q = lu.solve(rhs)
        q_forward.append(q.copy())
    q_forward = np.array(q_forward)

    # Propagate forward in s
    q = q_backward_init
    q_backward = [q.copy()]
    for n in range(Ns):
        rhs = M_right @ q
        q = lu.solve(rhs)
        q_backward.append(q.copy())
    q_backward = np.array(q_backward)

    # The partition function
    Q_p = np.trapezoid(q_forward[-1,:],z)

    # The polymer density
    rho_p = np.zeros_like(z)
    for n in range(Ns + 1):
        q_s = q_forward[n]
        qd_s = q_backward[Ns - n]
        rho_p += q_s * qd_s
    rho_p *= ds / Q_p
    rho_p *= sigma
    
    # Clip rho values to contrained values
    rho_p = np.where(rho_p >= 1.0, 1.0 - 1e-12, rho_p)
    
    # Solvent density and field
    rho_s = np.exp(-w_s)                       
    Q_s = np.trapezoid(rho_s, z) / V_s            
    rho_s /= Q_s   

    # Incompressibility residual
    residual = rho_p + rho_s - 1.0
    lambda_field += alpha * residual

    # Calculate new fields
    w_p_new = np.log(rho_p) + chi * rho_s + lambda_field
    w_s_new = np.log(rho_s) + chi * rho_p + lambda_field

    # Update field
    # Simple excluded volume model
    # w_new = chi * rho - np.log(1-rho)

    # Check convergence
    dw = np.linalg.norm(w_p_new - w_p) / np.linalg.norm(w_p_new)
    dw += np.linalg.norm(w_s_new - w_s) / np.linalg.norm(w_s_new)
    print(f"iter {iteration}: dw = {dw:.2e} mix = {mix:.3f} MSres = {np.sum(np.sqrt(residual**2)):.3f}")
    # print("w min:", w.min(), "w max:", w.max())

    if dw < tol:
        break

    # Adaptive mixing
    if prev_dw is not None:
        if dw > prev_dw:
            mix = max(mix * shrink, mix_min)
        else:
            mix = min(mix * grow, mix_max)

    # Update fields
    # w_p = mixer.mix(w_p, w_p_new)
    # w_s = mixer.mix(w_s, w_s_new)
    w_p = (1 - mix) * w_p + mix * w_p_new 
    w_s = (1 - mix) * w_s + mix * w_s_new

    prev_dw = dw

q0 = q_forward_init  # initial forward propagator
print("int q0(z) dz =", np.trapezoid(q_forward_init, z))
total_monomers = np.trapezoid(rho, z)
print("int rho(z) dz =", total_monomers)
print("Expected = sigma * N =", sigma * N)
print("Height = ", np.trapezoid(z*rho,z)/np.trapezoid(rho,z))

# -----------------
# Plot density
# -----------------
fig, ax = plt.subplots()
ax.plot(z, rho_p, "-", label='Polymer')
ax.plot(z, rho_s, "-", label="Solvent")
ax.set_ylabel('$\\rho$')
ax.set_xlabel('z')
ax.set_ylim([0,1])
ax.legend()
plt.show()