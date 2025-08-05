# polyDFT
Implementation of polymer density functional theory

## Project structure
polyDFT</br>
├ main.py 
    Entry point
├ config.py

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