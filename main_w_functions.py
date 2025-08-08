import numpy as np
import matplotlib.pyplot as plt

# System definition
Lz = 32
N = 32
b = 1.0
chi = 0
max_iter = 500
tol = 1e-6
mix = 0.2
sigma = 0.1
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

from src.dft.core import get_Lagrange_mat, get_Crank_Nicolson_mat, propagate_q, get_rho_p
w_p = np.zeros_like(z)
w_s = np.zeros_like(z)

DL = get_Lagrange_mat(Nz, dz, D)
M_right , lu = get_Crank_Nicolson_mat(w_p, DL, ds)
q_forward = propagate_q(q_forward_init, M_right, lu, Ns)
q_backward = propagate_q(q_backward_init, M_right, lu, Ns)
rho_p = get_rho_p(q_forward,q_backward,z,ds,Ns)

# print("H = ", np.trapz(rho_p*z,z)/np.trapz(rho_p,z))
# fig, ax = plt.subplots()
# ax.plot(z, rho_p, "-",label='polymer density')
# ax.set_ylabel('rho')
# ax.set_xlabel('z')
# ax.legend()
# plt.show()

from src.dft.solvers import Anderson, NewtonKrylovSCFT_Timed, SCFTStats
anderson = Anderson(m=5,beta=1.0,adaptive_beta=True)
stats = SCFTStats()
newton = NewtonKrylovSCFT_Timed(Nz=Nz, z=z, DL=DL, ds=ds, Ns=Ns,
                          qf_init=q_forward_init, qb_init=q_backward_init,
                          chi=chi, newton_tol=1e-6, max_newton=15, gmres_rtol=1e-2, 
                          ls_c=1e-4, ls_min=1e-3, restart=50, stats=stats, verbose=True)

lambda_field = np.zeros_like(z)
res_norms = []
dws = []

def print_scft_stats(s):
    import math

    # Totals
    print("\n=== Totals ===")
    print(f"Newton steps:         {len(s.gmres_iters)}")
    print(f"F(x) calls:           {s.F_calls}")
    print(f"Jv (matvec) calls:    {s.Jv_calls}")
    print(f"LU factorizations:    {s.LU_count}")
    print(f"Time total [s]:       {s.t_total:.3f}")
    if s.t_total > 0:
      print(f"  - F evals [s]:      {s.t_F:.3f}  ({100*s.t_F/s.t_total:5.1f}%)")
      print(f"  - GMRES [s]:        {s.t_gmres:.3f}  ({100*s.t_gmres/s.t_total:5.1f}%)")
      print(f"  - line search [s]:  {s.t_linesearch:.3f}  ({100*s.t_linesearch/s.t_total:5.1f}%)")
      print(f"  - LU [s]:           {s.t_lu:.3f}  ({100*s.t_lu/s.t_total:5.1f}%)")
      print(f"  - propagation [s]:  {s.t_prop:.3f}  ({100*s.t_prop/s.t_total:5.1f}%)")

    # Per-step table
    print("\n=== Per-Newton-step ===")
    hdr = ("step", "||F||/Nz", "gmres_it", "ls_halve", "alpha", "r1", "r2", "r3", "red_ratio")
    print("{:>4s} {:>10s} {:>8s} {:>8s} {:>6s} {:>10s} {:>10s} {:>10s} {:>10s}".format(*hdr))
    m = len(s.gmres_iters)
    for k in range(m):
        rn   = s.res_norms[k] if k < len(s.res_norms) else float("nan")
        gi   = s.gmres_iters[k]
        lsh  = s.ls_halvings[k]
        alp  = s.step_alpha[k]
        r1   = s.r1_norm[k]
        r2   = s.r2_norm[k]
        r3   = s.r3_norm[k]
        rr   = s.reduction_ratio[k]
        print(f"{k:4d} {rn:10.3e} {gi:8d} {lsh:8d} {alp:6.3f} {r1:10.3e} {r2:10.3e} {r3:10.3e} {rr:10.3e}")

(w_p, w_s, lambda_field), info = newton.solve(w_p, w_s, lambda_field)
print(info)
print_scft_stats(stats)

M_right , lu = get_Crank_Nicolson_mat(w_p, DL, ds)
q_forward = propagate_q(q_forward_init, M_right, lu, Ns)
q_backward = propagate_q(q_backward_init, M_right, lu, Ns)
rho_p = get_rho_p(q_forward,q_backward,z,ds,Ns,sigma)

# Clip rho values to contrained values
rho_p = np.where(rho_p >= 1.0, 1.0 - 1e-16, rho_p)
rho_s = np.exp(-w_s)                  

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

for iteration in range(max_iter):
    M_right , lu = get_Crank_Nicolson_mat(w_p, DL, ds)
    q_forward = propagate_q(q_forward_init, M_right, lu, Ns)
    q_backward = propagate_q(q_backward_init, M_right, lu, Ns)
    rho_p = get_rho_p(q_forward,q_backward,z,ds,Ns,sigma)

    # Clip rho values to contrained values
    rho_p = np.where(rho_p >= 1.0, 1.0 - 1e-16, rho_p)
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
    
    print(f"iter {iteration}: dw = {dw:.2e} res = {res_norm:.2e}")
    if dw < tol and res_norm < 1e-4:
        print(f"Converged in {iteration} iterations.")
        break
    
    
    # Update fields
    w_p = anderson.mix(w_p, w_p_new)
    w_s = anderson.mix(w_s, w_s_new)
    # w_p = (1 - mix) * w_p + mix * w_p_new 
    # w_s = (1 - mix) * w_s + mix * w_s_new

    # prev_dw = dw
    # mixes.append(mix)
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
