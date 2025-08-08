# polyDFT
Implementation of polymer density functional theory

## Project structure
```
polyDFT</br>
| main.py *Entry point* </br>
| README.md *Code explanation* </br>
├ docs </br>
  | polyDFT.md *Explanation of theory* </br>
├ src </br>
  ├ dft </br>
    | mixers.py *SCFT update schemes* </br>
```

## Conventions
| Mathematical Symbol | Meaning                         | Code Variable        |
| ------------------- | ------------------------------- | -------------------- |
| $\delta$            | Dirac delta (discretized)       | `delta_z`            |
| $\rho_P(z)$         | Monomer density                 | `rho_polymer`        |
| $w(z)$              | Self-consistent potential       | `potential_w`        |
| $q(z, s)$           | Forward propagator              | `prop_forward`       |
| $q^\dagger(z,s)$    | Backward propagator             | `prop_backward`      |
| $Q$                 | Single-chain partition function | `partition_function` |
| $\chi$              | Flory–Huggins parameter         | `chi`                |
| $z$                 | Spatial coordinate              | `z`                  |
| $s$                 | Contour variable along chain    | `s`                  |
| $N$                 | Chain length (segments)         | `N_segments`         |

## Propagators
A crank_nicolson propagator is implemented to solve the modified diffusion equation of the form:
$$\frac{\partial q(z,s)}{\partial s} = \frac{a^2}{6}\frac{\partial^2 q(z,s)}{\partial z^2} - w(z)q(z,s) $$
where $a$ is the Kuhn length.

We implement neumann boundary conditions (which cause oscillations).


## To Do
- [ ] Implement mirror system (probably also useful for electrostatics later)

