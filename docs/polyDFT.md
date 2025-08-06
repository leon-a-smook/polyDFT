# polyDFT
Polymer brushes consist of polymer chains that are densely grafted to a substrate. As a result, they stretch out and form a brush-like structure. One can predict properties of these systems using an extension of classical density functional theory for chain molecules (polyDFT). This project aims to implement this theory in a systematic and intuitive manner.

**Concepts**
- Polymer brush
- Chain propagators
- Crank-Nicolson scheme
- Self-consist field

## Polymer brush
A polymer brush consists of chains grafted to a substrate and are charaterized by a limited set of parameters.

| Parameter         | Symbol   | Variable name      | Unit              | 
| ---               | ---      | ---                | --                |
| Grafting density  | $\sigma$ | `sigma` | L<sup>-2</sup>    | 
| Chain length      | $N$      | `N`                | -                 |


## Chain propagators
Polymers are long molecules compared to solvent molecules such as water or methanol. As a result, their connectivity needs to be taken into account in polyDFT. This can be done using the Edwards diffusion equation which takes into account how a polymer chain would move in an external (generalized) potential $w(\mathbf{r})$. If we describe the chain as a continuous path along the contour variable $s$, we can describe the chain as a continuous path $\mathbf{r}(s)$.

If we also use the following definitions:
- $s \in \left[0,N\right]$ - contour variable
- $\mathbf{r}(s)$ - position of monomer (or segment) $s$
- $N$ - degree of polymerization
- $b$ - Kuhn length (or smallest length were contour tangent is uncorrelated with previous segments)

The statistical weight of a chain expressed like this consists of two parts:
1. the random walk (entropic cost of chain deformation)
2. energetic cost of placing a segment in the potential $w(\mathbf{r})$.

This gives a statistical weight that is propotional to
$$\mathit{P}\left[\mathbf{r}(s)\right] \propto \exp\left(-\frac{3}{2b^2}\int_0^N\left|\frac{d\mathbf{r}}{ds}\right|^2ds - \int_0^N w(\mathbf{r}(s))ds\right) $$

Next we define a propagator $q(\mathbf{r,s})$ such that it describes the total statistical weight of all chain configurations that start at a fixed initial position $\mathbf{r}_0$ and reach point $\mathbf{r}$ after a contour length $s$ in a potential landscape $w(\mathbf{r})$.

If you average over all starting positions $\mathbf{r}_0$, then $q(\mathbf{r},s)$ is the unconstrained chain propagator. For brushes we confine $\mathbf{r}_0$ to the grafting plane.

It can be shown that $q(\mathbf{r},s)$ satisfies the Edwards diffusion equation

$$\frac{\partial q(\mathbf{r},s)}{\partial s} = \frac{b^2}{6}\nabla^2 q(\mathbf{r},s) - w(\mathbf{r})q(\mathbf{r},s)$$

where the initial condition is set to $q(\mathbf{r},0) = \delta(\mathbf{r} -\mathbf{r}_0)$ for a brush.

The chain propagators enter into various thermodynamic quantities:

| Quantity              | Symbol    | Expression                                |
| ---                   | ---       | ---                                       |
| Partition function    | $Q$       |  $\int q(\mathbf{r},N) d\mathbf{r}$   |
| Monomer density       | $\rho(\mathbf{r})$ | $\frac{1}{Q}\int_0^N  q(\mathbf{r},s)q^\dag(\mathbf{r},N-s) ds$ |

## Solving Edwards diffusion equation in 1D
Given the diffusion equation
$$\frac{\partial q(z,s)}{\partial s} = D\frac{\partial^2 }{\partial z^2}q(z,s) - w(z)q(z,s)$$
where $D = b^2/6$. 

Additionally we define
- $q(z,0) = q_0(z)$
- $\frac{\partial q}{\partial z} = 0$ at $z = 0, L_z$ (no-flux Neumann boundary conditions)

We discretize $q(z,s)$ and solve with the Crank-Nicolson method.

| Dimension | Step | Points count | Grid |
| ---- | --- | --- | --- |
| $z$ | $\Delta z$ | $N_z = L_z / \Delta z$ | $z_i = i\Delta z$ for $i = 0, 1, ..., N_z - 1$ |
| $s$ | $\Delta s$ | $N_s = N / \Delta s$ | $s_n = n\Delta s$ for $n = 0, 1, ..., N_s$

Next we can define a Laplacian matrix since

$$\left.\frac{\partial^2 q}{\partial z^2}\right\rvert_{z = z_i} \approx \frac{q_{i+1}^n - 2q_i^n + q_{i-1}^n}{\Delta z^2} $$

can be entered into a matrix representation wiht matrix elements $L_{ij}$ where $\mathbf{\mathit{L}}\in \mathbb{R}^{N_z \times N_z}$

$$
L_{ij} = \begin{cases}
-2/\Delta z^2 &\mathrm{if} \quad i=j\\
1/\Delta z^2 &\mathrm{if} \quad |i-j| = 1\\
0 & \mathrm{otherwise}
\end{cases}
$$

To implement the boundary conditions, we adjust this matrix such that

- Dirichlet BCs: 
  - $L_{0,0} = 1.0$
  - $L_{N_z, N_z} = 1.0$
  - $q_0 = 0.0$
  - $q_{N_z} = 0.0$
- Neumann
  - $L_{0,0} = -2 / \Delta z^2$
  - $L_{0,1} = 2 / \Delta z^2$
  - $L_{N_z, N_z-1} = 2 / \Delta z^2$
  - $L_{N_z, N_z} = -2 / \Delta z^2$

## Optimized propagator solver
We can rewrite the diffusion equation as
$$\frac{\partial \mathbf{q}(s)}{\partial s} = (D\nabla^2 - \mathbf{W})\mathbf{q}(s)$$
which allows us to define the operator

$$\mathcal{A} = D\mathbf{L} - \mathbf{W}$$

here $\mathbf{W}$ is a diagonal matrix $\mathbf{W} = \text{diag}(w(z))$

The Crank-Nicolson update rule requires solving

$$\left(I - \frac{\Delta s}{2}\mathcal{A}\right) q^{n+1} = \left(I + \frac{\Delta s}{2}\mathcal{A}\right) q^{n}$$

in this expression $\mathbf{L}$ does not change if $\mathbf{W}$ is updated (e.g. in a self-consistent loop), so for efficiency, we do not want to recompute $D\mathbf{L}$. To exploit the fixed structure of $\mathbf{L}$, we
1. Precompute $D\mathbf{L}$
2. Update $\mathbf{W}$ when $w(z)$ changes
3. Update $\mathcal{A} = D\mathbf{L} - \mathbf{W}$ by only changing the values on the diagonal.

So to solve this, we construct the following matrices
- $M_{\text{left}} = I - \frac{\Delta s}{2}\mathcal{A}$
- $M_{\text{right}} = I + \frac{\Delta s}{2}\mathcal{A}$

and finally we solve

$$M_{\text{left}} \mathbf{q}^{n+1} = M_{\text{right}} \mathbf{q}^{n}$$

Since $M_{\text{left}}$ and $M_{\text{right}}$ are independent of $\mathbf{q}^n$, they can be reused for each iteration step over $n$.